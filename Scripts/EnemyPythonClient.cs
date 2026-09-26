using System;
using System.Collections;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class EnemyPythonClient : MonoBehaviour
{
    [Header("Python 서버")]
    public string host = "127.0.0.1";
    public int port = 5005;

    [Header("상대 = 플레이어 루트")]
    public Transform enemy;

    [Header("행동 설정")]
    public float decisionInterval = 0.05f;
    public float moveSpeed = 3f;
    public float maxRotationPerDecision = 30f;

    [Header("Episode")]
    public int maxEpisodeSteps = 1000; // Python train.py의 max_ep_len(현재 1000) 이하로 설정

    [Header("통신 안전장치")]
    public float actionTimeout = 0.3f;

    private Health myHealth;
    private Health enemyHealth;
    private Combat myCombat;
    private Combat enemyCombat;
    private Animator myAnimator;
    private Move enemyMove;
    private EnemyState stateRecorder;
    private CharacterController characterController;
    private Rigidbody rb;

    private Thread networkThread;
    private TcpClient client;
    private volatile bool running;
    private volatile bool connected;
    private readonly object dataLock = new object();

    private string pendingMessage;
    private bool pendingTerminal;
    private bool requestPending;
    private ActionMessage receivedAction;
    private bool hasReceivedAction;
    private bool hasEpisodeEndAck;
    private string networkError;

    private int currentMovement;
    private float lastActionTime;
    private float decisionTimer;
    private int episodeStep;
    private bool episodeActive;
    private bool waitingForEpisodeEnd;
    private bool readyForReset = true;

    // AI의 직전 관측 구간 행동
    private int myPreviousMovement;
    private float accumulatedMyRotation;
    private int accumulatedMyCombat;

    [Serializable]
    private class StateMessage
    {
        public Vector3 enemyRelativePosition;
        public float distance;
        public float angleToEnemy;
        public float myHealth;
        public float enemyHealth;
        public int myAction;
        public int enemyAction;
        public int myPreviousMovement;
        public float myPreviousRotation;
        public int myPreviousCombat;
        public int enemyPreviousMovement;
        public float enemyPreviousRotation;
        public int enemyPreviousCombat;
    }

    // 기존 state를 중첩해 Python UnityTCPEnv의 reset/step 메시지와 맞춘다.
    [Serializable]
    private class ObservationMessage
    {
        public string type;
        public StateMessage state;
        public float reward;
        public bool done;
    }

    [Serializable]
    private class ActionMessage
    {
        public string type;
        public int movement; // 0 Idle, 1 W, 2 A, 3 S, 4 D
        public float rotation; // 도 단위
        public int combat; // 0 Idle, 1 Attack, 2 Parry
    }

    [Serializable]
    private class ControlMessage
    {
        public string type;
    }

    public void CapturePreviousAction(out int movement, out float rotation, out int combat)
    {
        movement = myPreviousMovement;
        rotation = accumulatedMyRotation;
        combat = accumulatedMyCombat;
        accumulatedMyRotation = 0f;
        accumulatedMyCombat = 0;
    }

    private void Awake()
    {
        myHealth = GetComponent<Health>();
        myCombat = GetComponent<Combat>();
        stateRecorder = GetComponent<EnemyState>();
        myAnimator = GetComponent<Animator>();
        characterController = GetComponent<CharacterController>();
        rb = GetComponent<Rigidbody>();

        if (enemy != null)
        {
            enemyHealth = enemy.GetComponent<Health>();
            enemyCombat = enemy.GetComponent<Combat>();
            enemyMove = enemy.GetComponent<Move>();
        }
    }

    private void Start()
    {
        if (enemy == null || myHealth == null || enemyHealth == null ||
            myCombat == null || enemyCombat == null || myAnimator == null ||
            enemyMove == null || stateRecorder == null)
        {
            Debug.LogError("[EnemyPythonClient] 참조 오류: enemy, 양쪽 Health/Combat, AI Animator, 플레이어 Move, AI EnemyState 확인");
            enabled = false;
            return;
        }

        // Health의 2초 자동 부활 대신, Python의 episode_end 확인 후 양쪽을 같이 리셋한다.
        myHealth.externallyManagedRespawn = true;
        enemyHealth.externallyManagedRespawn = true;

        // 첫 reset/action 이전에는 전투 및 플레이어 입력을 잠시 멈춘다.
        // Python train/server 둘 다 첫 reset에 action으로 응답한다.
        myCombat.enabled = false;
        enemyCombat.enabled = false;
        enemyMove.enabled = false;

        running = true;
        lastActionTime = Time.time;
        networkThread = new Thread(NetworkLoop) { IsBackground = true };
        networkThread.Start();
    }

    private void FixedUpdate()
    {
        ActionMessage newAction = null;
        bool endAck = false;
        string error = null;

        lock (dataLock)
        {
            if (hasReceivedAction)
            {
                newAction = receivedAction;
                receivedAction = null;
                hasReceivedAction = false;
            }

            if (hasEpisodeEndAck)
            {
                endAck = true;
                hasEpisodeEndAck = false;
            }

            if (networkError != null)
            {
                error = networkError;
                networkError = null;
            }
        }

        if (error != null)
            Debug.LogError("[EnemyPythonClient] " + error);

        if (!connected)
        {
            // 서버가 종료되면 이전 action으로 계속 싸우지 않도록 중지.
            // 모드 변경 시 새 Python 프로세스를 실행한 뒤 Unity Play를 다시 시작한다.
            currentMovement = 0;
            myCombat.enabled = false;
            enemyCombat.enabled = false;
            enemyMove.enabled = false;
            return;
        }

        if (waitingForEpisodeEnd)
        {
            currentMovement = 0;
            if (endAck)
                ResetBothCharacters();
            return;
        }

        // 새 판의 초기 상태는 reset 메시지로 한 번만 보낸다.
        if (!episodeActive)
        {
            if (readyForReset && !requestPending)
            {
                episodeStep = 0;
                decisionTimer = 0f;

                if (QueueObservation("reset", 0f, false))
                    episodeActive = true;
            }
            return;
        }

        if (newAction != null)
        {
            // Python update 중에는 양쪽을 멈춰 두었다가 새 판의 첫 action에서 재개.
            if (episodeStep == 0)
            {
                myCombat.enabled = true;
                enemyCombat.enabled = true;
                enemyMove.enabled = true;
            }

            ApplyNewAction(newAction);
            episodeStep++; // Python에서 실제로 선택한 행동 하나를 1 step으로 센다.
            decisionTimer = 0f;
            lastActionTime = Time.time;
        }

        // reset 또는 step에 대한 action이 아직 도착하지 않았으면 기다린다.
        if (episodeStep == 0 || requestPending)
            return;

        if (Time.time - lastActionTime > actionTimeout)
            currentMovement = 0;

        if (myHealth.currentHealth > 0 && enemyHealth.currentHealth > 0)
            ApplyMovement(currentMovement);
        else
            currentMovement = 0;

        decisionTimer += Time.fixedDeltaTime;

        bool healthTerminal =
            myHealth.currentHealth <= 0 || enemyHealth.currentHealth <= 0;
        bool limitTerminal = episodeStep >= maxEpisodeSteps;
        bool done = healthTerminal || limitTerminal;

        if (!done && decisionTimer < decisionInterval)
            return;

        // 사망 이벤트는 프레임 끝에서 trajectory를 확정한다.
        // 같은 타격으로 발생하는 OnAttackHit까지 포함한 뒤 마지막 reward를 전송한다.
        if (healthTerminal && stateRecorder.IsFinalizingTrajectory)
        {
            currentMovement = 0;
            return;
        }

        if (limitTerminal && !healthTerminal)
            stateRecorder.FinishTrajectoryForStepLimit();

        if (!QueueObservation("step", 0f, done))
            return;

        // 이미 결과를 보고한 행동이 서버 응답 대기 중에 더 실행되지 않도록 정지한다.
        currentMovement = 0;
        decisionTimer = 0f;

        if (done)
        {
            waitingForEpisodeEnd = true;
            myCombat.enabled = false;
            enemyCombat.enabled = false;
            enemyMove.enabled = false;
            Debug.Log("[EnemyPythonClient] Episode 끝: steps=" + episodeStep +
                      ", reason=" + (healthTerminal ? "health" : "maxEpisodeSteps"));
        }
    }

    private bool QueueObservation(string type, float reward, bool done)
    {
        StateMessage state = BuildState();
        if (state == null)
            return false;

        ObservationMessage message = new ObservationMessage
        {
            type = type,
            state = state,
            reward = reward,
            done = done
        };

        lock (dataLock)
        {
            if (requestPending)
                return false;

            if (type == "step")
                message.reward = stateRecorder.ConsumeRewardForPython();

            pendingMessage = JsonUtility.ToJson(message);
            pendingTerminal = done;
            requestPending = true;
        }
        return true;
    }

    private StateMessage BuildState()
    {
        if (stateRecorder == null || stateRecorder.LatestState == null)
            return null;

        EnemyState.StateData recorded = stateRecorder.LatestState;
        StateMessage state = new StateMessage();
        state.enemyRelativePosition = recorded.enemyRelativePosition;
        state.distance = recorded.distance;
        state.angleToEnemy = recorded.angleToEnemy;
        state.myHealth = recorded.myHealth;
        state.enemyHealth = recorded.enemyHealth;
        state.myAction = recorded.myAction;
        state.enemyAction = recorded.enemyAction;
        state.myPreviousMovement = recorded.myPreviousMovement;
        state.myPreviousRotation = recorded.myPreviousRotation;
        state.myPreviousCombat = recorded.myPreviousCombat;
        state.enemyPreviousMovement = recorded.enemyPreviousMovement;
        state.enemyPreviousRotation = recorded.enemyPreviousRotation;
        state.enemyPreviousCombat = recorded.enemyPreviousCombat;
        return state;
    }

    private void ResetBothCharacters()
    {
        // Python train.py는 PPO.update()와 latest.pth 저장이 끝난 다음에만
        // episode_end를 보낸다. server.py는 학습 없이 즉시 보낸다.
        // 이 메서드는 반드시 메인 스레드에서만 실행한다.
        currentMovement = 0;
        myPreviousMovement = 0;
        accumulatedMyRotation = 0f;
        accumulatedMyCombat = 0;
        episodeStep = 0;
        decisionTimer = 0f;
        receivedAction = null;
        hasReceivedAction = false;

        stateRecorder.SetTrajectoryRestartDeferred(true);
        try
        {
            myHealth.ResetForNewEpisode();
            enemyHealth.ResetForNewEpisode();

            // 다음 판 첫 관측값에 이전 판의 플레이어 행동이 남지 않도록 초기화.
            enemyMove.ResetPreviousAction();
        }
        finally
        {
            stateRecorder.StartTrajectoryAfterEpisodeReset();
        }

        // 새 reset/action을 받을 때까지 두 캐릭터를 대기시킨다.
        myCombat.enabled = false;
        enemyCombat.enabled = false;
        enemyMove.enabled = false;

        waitingForEpisodeEnd = false;
        episodeActive = false;
        readyForReset = false;
        lastActionTime = Time.time;

        StartCoroutine(AllowNextResetAfterFrame());
        Debug.Log("[EnemyPythonClient] episode_end 수신 → 양쪽 리셋 완료");
    }

    private IEnumerator AllowNextResetAfterFrame()
    {
        yield return new WaitForEndOfFrame();
        readyForReset = true;
    }

    private void ApplyNewAction(ActionMessage action)
    {
        if (myHealth.currentHealth <= 0 || enemyHealth.currentHealth <= 0)
        {
            currentMovement = 0;
            return;
        }

        currentMovement = Mathf.Clamp(action.movement, 0, 4);
        myPreviousMovement = currentMovement;

        float rotation = Mathf.Clamp(
            action.rotation, -maxRotationPerDecision, maxRotationPerDecision);

        if (rb != null && !rb.isKinematic)
            rb.MoveRotation(rb.rotation * Quaternion.Euler(0f, rotation, 0f));
        else
            transform.Rotate(0f, rotation, 0f, Space.Self);

        accumulatedMyRotation += rotation;

        int combatCommand = Mathf.Clamp(action.combat, 0, 2);
        if (combatCommand != 0)
            accumulatedMyCombat = combatCommand;

        if (myCombat.CurrentAction != Combat.ActionState.Idle)
            return;

        switch (combatCommand)
        {
            case 1:
                myCombat.Attack();
                myAnimator.SetTrigger("Attack");
                break;
            case 2:
                myCombat.Parry();
                myAnimator.SetTrigger("Defense");
                break;
        }
    }

    private void ApplyMovement(int movement)
    {
        Vector3 localDirection = Vector3.zero;
        switch (movement)
        {
            case 1: localDirection = Vector3.forward; break;
            case 2: localDirection = Vector3.left; break;
            case 3: localDirection = Vector3.back; break;
            case 4: localDirection = Vector3.right; break;
        }
        if (localDirection == Vector3.zero)
            return;

        Vector3 worldDirection = transform.TransformDirection(localDirection);
        worldDirection.y = 0f;
        worldDirection.Normalize();
        Vector3 displacement = worldDirection * moveSpeed * Time.fixedDeltaTime;

        if (characterController != null && characterController.enabled)
            characterController.Move(displacement);
        else if (rb != null && !rb.isKinematic)
            rb.MovePosition(rb.position + displacement);
        else
            transform.position += displacement;
    }

    private void NetworkLoop()
    {
        StreamReader reader = null;
        StreamWriter writer = null;
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            NetworkStream stream = client.GetStream();
            // PPO 학습 시간 때문에 5초 읽기 제한을 두지 않는다. 종료 시 client.Close()로 해제.
            stream.ReadTimeout = Timeout.Infinite;
            stream.WriteTimeout = 5000;
            reader = new StreamReader(stream, new UTF8Encoding(false));
            writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
            connected = true;

            while (running)
            {
                string toSend = null;
                bool terminal = false;
                lock (dataLock)
                {
                    if (requestPending && pendingMessage != null)
                    {
                        toSend = pendingMessage;
                        terminal = pendingTerminal;
                        pendingMessage = null;
                    }
                }

                if (toSend == null)
                {
                    Thread.Sleep(1);
                    continue;
                }

                writer.WriteLine(toSend);
                string response = reader.ReadLine();
                if (response == null)
                    break;

                if (terminal)
                {
                    ControlMessage ack = JsonUtility.FromJson<ControlMessage>(response);
                    if (ack == null || ack.type != "episode_end")
                        throw new Exception("Expected episode_end, received: " + response);
                    lock (dataLock)
                    {
                        hasEpisodeEndAck = true;
                        requestPending = false;
                    }
                }
                else
                {
                    ActionMessage action = JsonUtility.FromJson<ActionMessage>(response);
                    if (action == null || (action.type != null && action.type != "action"))
                        throw new Exception("Action JSON 파싱 실패: " + response);
                    lock (dataLock)
                    {
                        receivedAction = action;
                        hasReceivedAction = true;
                        requestPending = false;
                    }
                }
            }
        }
        catch (Exception e)
        {
            if (running)
            {
                lock (dataLock)
                    networkError = e.ToString();
            }
        }
        finally
        {
            connected = false;
            lock (dataLock)
            {
                requestPending = false;
                pendingMessage = null;
            }
            try { reader?.Close(); } catch { }
            try { writer?.Close(); } catch { }
            try { client?.Close(); } catch { }
        }
    }

    private void OnDisable()
    {
        running = false;
        connected = false;
        try { client?.Close(); } catch { }
        if (networkThread != null && networkThread.IsAlive &&
            Thread.CurrentThread != networkThread)
            networkThread.Join(500);
    }

    private void OnDestroy()
    {
        running = false;
        try { client?.Close(); } catch { }
    }
}
