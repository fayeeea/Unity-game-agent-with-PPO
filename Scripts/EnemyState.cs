using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class EnemyState : MonoBehaviour
{
    [Header("AI의 상대 = 플레이어 루트")]
    public Transform enemy;

    [Header("현재 State 확인용")]
    // AI 자신의 월드 좌표 (상대 위치 대신 사용)
    public Vector3 myPosition;
    public float distance;
    public float angleToEnemy;

    public float myHealth;
    public float enemyHealth;

    public Combat.ActionState myAction;
    public Combat.ActionState enemyAction;

    [Header("이전 행동 확인용")]
    public int myPreviousMovement;
    public float myPreviousRotation;
    public int myPreviousCombat;

    public int enemyPreviousMovement;
    public float enemyPreviousRotation;
    public int enemyPreviousCombat;

    private const float SAMPLE_INTERVAL = 0.05f;

    private Health myHealthComponent;
    private Health enemyHealthComponent;

    private Combat myCombat;
    private Combat enemyCombat;

    private Move enemyMove;
    private EnemyPythonClient pythonClient;

    private Coroutine collectCoroutine;
    private Trajectory currentTrajectory;
    private int trajectoryNumber;

    [Header("Reward (AI 기준, 전투 이벤트 1회당)")]
    public float rewardAttackHit = 2f;
    public float rewardAttackMissed = -0.2f;
    public float rewardParrySuccess = 2f;
    public float rewardDamaged = -1f;
    public float rewardMyDeath = -10f;
    public float rewardEnemyDeath = 10f;

    [Header("Reward (상대에게 등을 보일 때, 샘플 1회당)")]
    [Range(90f, 180f)] public float facingAwayAngle = 120f;
    public float rewardFacingAway = -0.01f;

    [Header("Reward (상대와의 거리 변화, 샘플 1회당)")]
    public float preferredCombatDistance = 1.5f;
    public float minDistanceChange = 0.02f;
    public float rewardApproach = 0.02f;
    public float rewardRetreat = -0.03f;

    private float previousDistance;
    private bool hasPreviousDistance;

    // 마지막 샘플 이후 발생한 보상을 종류별로 누적한다.
    private float pendingAttackHitReward;
    private float pendingAttackMissReward;
    private float pendingParryReward;
    private float pendingDamagedReward;
    private float pendingDeathReward;
    private float pendingFacingAwayReward;
    private float pendingDistanceReward;

    // Trajectory 샘플링과 독립적인 PPO 보상 누적기.
    // 0.05초마다 샘플링해도 Python이 아직 요청하지 않은 reward가 사라지지 않는다.
    private float pendingPythonReward;

    // 사망 처리 중에는 OnAttackHit 등 같은 타격의 후속 이벤트까지 수집한다.
    public bool IsFinalizingTrajectory => finishTrajectoryCoroutine != null;

    private bool myDied;
    private bool enemyDied;
    private Coroutine finishTrajectoryCoroutine;

    // Python 통신 스크립트가 읽을 최신 관측값
    public StateData LatestState { get; private set; }

    [Serializable]
    public class StateData
    {
        public float time;

        // 직전 샘플 이후 발생한 보상. 전투 이벤트가 없으면 모두 0.
        public float reward;
        public float attackHitReward;
        public float attackMissReward;
        public float parryReward;
        public float damagedReward;
        public float deathReward;
        public float facingAwayReward;
        public float distanceReward;

        // 통신 호환성: 기존 JSON 키는 유지하지만 값은 AI 자신의 월드 좌표.
        public Vector3 enemyRelativePosition;
        public float distance;
        public float angleToEnemy;

        public float myHealth;
        public float enemyHealth;

        // 현재 실제 전투 상태
        public int myAction;
        public int enemyAction;

        // AI의 이전 행동
        public int myPreviousMovement;
        public float myPreviousRotation;
        public int myPreviousCombat;

        // 플레이어의 이전 행동
        public int enemyPreviousMovement;
        public float enemyPreviousRotation;
        public int enemyPreviousCombat;
    }

    [Serializable]
    public class Trajectory
    {
        public int trajectoryNumber;
        public float sampleInterval;
        public string endReason;

        public float totalReward;

        public List<StateData> states = new List<StateData>();
    }

    private void Start()
    {
        myHealthComponent = GetComponent<Health>();
        myCombat = GetComponent<Combat>();
        pythonClient = GetComponent<EnemyPythonClient>();

        if (enemy != null)
        {
            enemyHealthComponent = enemy.GetComponent<Health>();
            enemyCombat = enemy.GetComponent<Combat>();
            enemyMove = enemy.GetComponent<Move>();
        }

        if (enemy == null ||
            myHealthComponent == null ||
            enemyHealthComponent == null ||
            myCombat == null ||
            enemyCombat == null ||
            enemyMove == null)
        {
            Debug.LogError(
                "[EnemyState] 참조 오류. AI와 플레이어 루트의 " +
                "Health/Combat, 플레이어 Move, Enemy 연결을 확인하세요."
            );

            enabled = false;
            return;
        }

        myHealthComponent.OnDied += HandleMyDeath;
        myHealthComponent.OnRespawned += HandleRespawn;
        enemyHealthComponent.OnDied += HandleEnemyDeath;
        enemyHealthComponent.OnRespawned += HandleRespawn;

        // 현재 AI의 전투 결과만 구독한다. 상대의 공격 성공은
        // AI의 OnDamaged로 이미 계산하므로 이중으로 세지 않는다.
        myCombat.OnAttackHit += HandleAttackHit;
        myCombat.OnAttackMissed += HandleAttackMissed;
        myHealthComponent.OnParrySuccess += HandleParrySuccess;
        myHealthComponent.OnDamaged += HandleDamaged;

        StartNewTrajectory();
    }

    private void OnDestroy()
    {
        if (myHealthComponent != null)
        {
            myHealthComponent.OnDied -= HandleMyDeath;
            myHealthComponent.OnRespawned -= HandleRespawn;
            myHealthComponent.OnParrySuccess -= HandleParrySuccess;
            myHealthComponent.OnDamaged -= HandleDamaged;
        }

        if (enemyHealthComponent != null)
        {
            enemyHealthComponent.OnDied -= HandleEnemyDeath;
            enemyHealthComponent.OnRespawned -= HandleRespawn;
        }

        if (myCombat != null)
        {
            myCombat.OnAttackHit -= HandleAttackHit;
            myCombat.OnAttackMissed -= HandleAttackMissed;
        }
    }

    private void HandleAttackHit()
    {
        pendingAttackHitReward += rewardAttackHit;
        pendingPythonReward += rewardAttackHit;
    }

    private void HandleAttackMissed()
    {
        pendingAttackMissReward += rewardAttackMissed;
        pendingPythonReward += rewardAttackMissed;
    }

    private void HandleParrySuccess()
    {
        pendingParryReward += rewardParrySuccess;
        pendingPythonReward += rewardParrySuccess;
    }

    private void HandleDamaged(int actualDamage)
    {
        if (actualDamage > 0)
        {
            pendingDamagedReward += rewardDamaged;
            pendingPythonReward += rewardDamaged;
        }
    }

    // Python의 step 메시지가 실제로 큐에 들어갈 때만 호출한다.
    // Trajectory용 reward와는 별개이며 한 번 소비하면 0으로 초기화된다.
    public float ConsumeRewardForPython()
    {
        float reward = pendingPythonReward;
        pendingPythonReward = 0f;
        return reward;
    }

    private void ClearPendingRewards()
    {
        pendingAttackHitReward = 0f;
        pendingAttackMissReward = 0f;
        pendingParryReward = 0f;
        pendingDamagedReward = 0f;
        pendingDeathReward = 0f;
        pendingFacingAwayReward = 0f;
        pendingDistanceReward = 0f;
    }

    private void StartNewTrajectory()
    {
        if (collectCoroutine != null)
        {
            StopCoroutine(collectCoroutine);
            collectCoroutine = null;
        }

        trajectoryNumber++;

        currentTrajectory = new Trajectory
        {
            trajectoryNumber = trajectoryNumber,
            sampleInterval = SAMPLE_INTERVAL,
            endReason = "",
            totalReward = 0f
        };

        LatestState = null;
        myDied = false;
        enemyDied = false;
        ClearPendingRewards();
        pendingPythonReward = 0f;
        previousDistance = 0f;
        hasPreviousDistance = false;

        collectCoroutine = StartCoroutine(CollectRoutine());

        Debug.Log(
            "[EnemyState] Trajectory " +
            trajectoryNumber +
            " 수집 시작"
        );
    }

    private IEnumerator CollectRoutine()
    {
        while (true)
        {
            CollectState();
            yield return new WaitForSeconds(SAMPLE_INTERVAL);
        }
    }

    public void CollectState()
    {
        if (enemy == null || currentTrajectory == null)
            return;

        // 기존 상대 로컬 위치 대신 AI 자신의 월드 절대 위치를 기록한다.
        myPosition = transform.position;

        Vector3 direction =
            enemy.position - transform.position;

        direction.y = 0f;

        distance = direction.magnitude;

        // 전투 거리 밖에서 접근하면 보너스, 더 멀어지면 패널티.
        // 첫 관측값에는 이전 거리가 없으므로 보상을 주지 않는다.
        if (hasPreviousDistance &&
            myHealthComponent.currentHealth > 0 &&
            enemyHealthComponent.currentHealth > 0 &&
            distance > preferredCombatDistance)
        {
            float distanceChange = previousDistance - distance;
            float distanceReward = 0f;

            if (distanceChange > minDistanceChange)
                distanceReward = rewardApproach;
            else if (distanceChange < -minDistanceChange)
                distanceReward = rewardRetreat;

            pendingDistanceReward += distanceReward;
            pendingPythonReward += distanceReward;
        }

        previousDistance = distance;
        hasPreviousDistance = true;

        Vector3 forward = transform.forward;
        forward.y = 0f;

        angleToEnemy = Vector3.SignedAngle(
            forward,
            direction,
            Vector3.up
        );

        // AI가 살아 있고 상대에게 등을 보이는 동안, 관측 샘플마다 작은 패널티.
        // signed angle은 좌/우에 따라 부호가 달라지므로 절댓값으로 판정한다.
        if (myHealthComponent.currentHealth > 0 &&
            enemyHealthComponent.currentHealth > 0 &&
            Mathf.Abs(angleToEnemy) >= facingAwayAngle)
        {
            pendingFacingAwayReward += rewardFacingAway;
            pendingPythonReward += rewardFacingAway;
        }

        // 체력
        myHealth =
            (float)myHealthComponent.currentHealth /
            Mathf.Max(1, myHealthComponent.maxHealth);

        enemyHealth =
            (float)enemyHealthComponent.currentHealth /
            Mathf.Max(1, enemyHealthComponent.maxHealth);

        // 현재 실제 전투 상태
        myAction = myCombat.CurrentAction;
        enemyAction = enemyCombat.CurrentAction;

        // AI 이전 행동: Python 통신 스크립트가 기록한 값
        if (pythonClient != null)
        {
            pythonClient.CapturePreviousAction(
                out myPreviousMovement,
                out myPreviousRotation,
                out myPreviousCombat
            );
        }
        else
        {
            myPreviousMovement = 0;
            myPreviousRotation = 0f;
            myPreviousCombat = 0;
        }

        // 플레이어 이전 행동: 여기서 단 한 번만 확정
        enemyMove.CapturePreviousAction();

        enemyPreviousMovement = enemyMove.PreviousMovement;
        enemyPreviousRotation = enemyMove.PreviousRotation;
        enemyPreviousCombat = enemyMove.PreviousCombat;

        StateData state = new StateData
        {
            time = currentTrajectory.states.Count * SAMPLE_INTERVAL,

            attackHitReward = pendingAttackHitReward,
            attackMissReward = pendingAttackMissReward,
            parryReward = pendingParryReward,
            damagedReward = pendingDamagedReward,
            deathReward = pendingDeathReward,
            facingAwayReward = pendingFacingAwayReward,
            distanceReward = pendingDistanceReward,
            reward = pendingAttackHitReward + pendingAttackMissReward
                   + pendingParryReward + pendingDamagedReward + pendingDeathReward
                   + pendingFacingAwayReward + pendingDistanceReward,

            enemyRelativePosition = myPosition, // 기존 Python 31차원 전처리와 호환
            distance = distance,
            angleToEnemy = angleToEnemy,

            myHealth = myHealth,
            enemyHealth = enemyHealth,

            myAction = (int)myAction,
            enemyAction = (int)enemyAction,

            myPreviousMovement = myPreviousMovement,
            myPreviousRotation = myPreviousRotation,
            myPreviousCombat = myPreviousCombat,

            enemyPreviousMovement = enemyPreviousMovement,
            enemyPreviousRotation = enemyPreviousRotation,
            enemyPreviousCombat = enemyPreviousCombat
        };

        currentTrajectory.totalReward += state.reward;
        ClearPendingRewards();

        LatestState = state;
        currentTrajectory.states.Add(state);
    }

    private void HandleMyDeath()
    {
        if (currentTrajectory == null) return;

        pendingDeathReward += rewardMyDeath;
        pendingPythonReward += rewardMyDeath;
        myDied = true;
        ScheduleTrajectoryFinish();
    }

    private void HandleEnemyDeath()
    {
        if (currentTrajectory == null) return;

        pendingDeathReward += rewardEnemyDeath;
        pendingPythonReward += rewardEnemyDeath;
        enemyDied = true;
        ScheduleTrajectoryFinish();
    }

    private void ScheduleTrajectoryFinish()
    {
        if (finishTrajectoryCoroutine != null) return;

        if (collectCoroutine != null)
        {
            StopCoroutine(collectCoroutine);
            collectCoroutine = null;
        }

        // OnDied는 TakeDamage 중에 발생하므로, 같은 타격의 OnAttackHit까지
        // 수집한 뒤 마지막 state를 기록한다.
        finishTrajectoryCoroutine = StartCoroutine(FinishTrajectoryAtEndOfFrame());
    }

    private IEnumerator FinishTrajectoryAtEndOfFrame()
    {
        yield return new WaitForEndOfFrame();

        if (currentTrajectory != null)
        {
            CollectState();

            string reason = myDied && enemyDied ? "Both_Died"
                : myDied ? "AI_Died" : "Player_Died";
            SaveTrajectory(reason);
            currentTrajectory = null;
        }

        finishTrajectoryCoroutine = null;
    }

    // 체력 사망이 아닌 decision 제한으로 판이 종료될 때 호출한다.
    public void FinishTrajectoryForStepLimit()
    {
        if (currentTrajectory == null || finishTrajectoryCoroutine != null)
            return;

        if (collectCoroutine != null)
        {
            StopCoroutine(collectCoroutine);
            collectCoroutine = null;
        }

        CollectState();
        SaveTrajectory("Step_Limit");
        currentTrajectory = null;
    }

    // Python의 episode_end 응답 후 두 캐릭터를 리셋하는 동안 자동 시작 방지.
    private bool deferRespawnStart;

    public void SetTrajectoryRestartDeferred(bool value)
    {
        deferRespawnStart = value;
    }

    public void StartTrajectoryAfterEpisodeReset()
    {
        deferRespawnStart = false;
        StartNewTrajectory();
    }

    private void HandleRespawn()
    {
        if (deferRespawnStart)
            return;
        if (currentTrajectory != null || finishTrajectoryCoroutine != null)
            return;

        // 둘 다 살아 있을 때에만 다음 trajectory를 시작한다.
        if (myHealthComponent.currentHealth <= 0 ||
            enemyHealthComponent.currentHealth <= 0)
            return;

        StartNewTrajectory();
    }

    private void SaveTrajectory(string reason)
    {
        if (currentTrajectory == null)
            return;

        currentTrajectory.endReason = reason;

        string folder = Path.Combine(
            Application.dataPath,
            "Trajectories"
        );

        Directory.CreateDirectory(folder);

        // 매 판 종료 후 같은 이름을 덮어써서 최신 완료 trajectory 하나만 유지한다.
        string fullPath = Path.Combine(folder, "latest_trajectory.json");

        string json = JsonUtility.ToJson(
            currentTrajectory,
            true
        );

        File.WriteAllText(fullPath, json);

        Debug.Log(
            "[EnemyState] Trajectory 저장 완료!" +
            "\n번호: " + currentTrajectory.trajectoryNumber +
            "\n샘플 수: " + currentTrajectory.states.Count +
            "\n경로: " + fullPath
        );

#if UNITY_EDITOR
        UnityEditor.AssetDatabase.Refresh();
#endif
    }
}