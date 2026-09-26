using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Combat : MonoBehaviour
{
    public enum ActionState
    {
        Idle = 0,
        Attack = 1,
        Parry = 2
    }

    public ActionState CurrentAction { get; private set; } = ActionState.Idle;

    private const int ATTACK_DAMAGE = 20;
    private const float ATTACK_RANGE = 0.7f;
    private const float ATTACK_RADIUS = 1f;
    private const float ATTACK_HALF_ANGLE = 75f; // 정면 기준 좌우 60도 (총 120도)
    private const float ATTACK_DELAY = 6f / 24f;
    private const float ATTACK_COOLDOWN = 12f / 24f;

    private const float PARRY_START = 4f / 24f;
    private const float PARRY_END = 8f / 24f;
    private const float PARRY_COOLDOWN = 12f / 24f;

    [Header("Attack Position")]
    public Transform attackPoint;

    // Reward 연결용: 공격 1회에 결과 이벤트 1개만 발생
    public event Action OnAttackHit;
    public event Action OnAttackMissed;
    public event Action OnAttackParried;

    private Health health;
    private bool canAttack = true;
    private bool canParry = true;

    private Coroutine attackCoroutine;
    private Coroutine parryCoroutine;

    void Awake()
    {
        health = GetComponent<Health>();
    }

    public void Attack()
    {
        if (!isActiveAndEnabled ||
            !canAttack ||
            CurrentAction != ActionState.Idle ||
            health == null ||
            health.currentHealth <= 0)
        {
            return;
        }

        CurrentAction = ActionState.Attack;
        attackCoroutine = StartCoroutine(AttackRoutine());
    }

    private IEnumerator AttackRoutine()
    {
        canAttack = false;

        yield return new WaitForSeconds(ATTACK_DELAY);

        Vector3 origin = attackPoint != null
            ? attackPoint.position
            : transform.position + Vector3.up * 1.2f;

        RaycastHit[] hits = Physics.SphereCastAll(
            origin,
            ATTACK_RADIUS,
            transform.forward,
            ATTACK_RANGE
        );

        HashSet<Health> checkedTargets = new HashSet<Health>();
        bool dealtDamage = false;
        bool wasParried = false;

        foreach (RaycastHit hit in hits)
        {
            Health target = hit.collider.GetComponentInParent<Health>();

            if (target == null || target == health)
                continue;

            // 구체 판정에 걸려도 캐릭터 뒤쪽에 있는 대상은 공격하지 않음
            Vector3 directionToTarget = target.transform.position - transform.position;
            directionToTarget.y = 0f;

            Vector3 forward = transform.forward;
            forward.y = 0f;

            if (directionToTarget.sqrMagnitude < 0.0001f ||
                Vector3.Angle(forward, directionToTarget) > ATTACK_HALF_ANGLE)
                continue;

            if (!checkedTargets.Add(target))
                continue;

            Health.DamageResult result = target.ApplyDamage(ATTACK_DAMAGE);

            if (result == Health.DamageResult.Damaged)
                dealtDamage = true;
            else if (result == Health.DamageResult.Parried)
                wasParried = true;
        }

        // 명령을 받은 순간이 아니라 실제 공격 판정 시점에 결과를 알림
        if (dealtDamage)
            OnAttackHit?.Invoke();
        else if (wasParried)
            OnAttackParried?.Invoke();
        else
            OnAttackMissed?.Invoke();

        yield return new WaitForSeconds(
            Mathf.Max(0f, ATTACK_COOLDOWN - ATTACK_DELAY)
        );

        canAttack = true;
        CurrentAction = ActionState.Idle;
        attackCoroutine = null;
    }

    public void Parry()
    {
        if (!isActiveAndEnabled ||
            !canParry ||
            CurrentAction != ActionState.Idle ||
            health == null ||
            health.currentHealth <= 0)
        {
            return;
        }

        CurrentAction = ActionState.Parry;
        parryCoroutine = StartCoroutine(ParryRoutine());
    }

    private IEnumerator ParryRoutine()
    {
        canParry = false;

        // 준비 동작
        yield return new WaitForSeconds(PARRY_START);

        // 실제 패링 판정 구간
        health.SetParrying(true);

        yield return new WaitForSeconds(PARRY_END - PARRY_START);

        health.SetParrying(false);

        // 후딜레이
        yield return new WaitForSeconds(PARRY_COOLDOWN - PARRY_END);

        canParry = true;
        CurrentAction = ActionState.Idle;
        parryCoroutine = null;
    }

    private void OnDisable()
    {
        if (attackCoroutine != null)
        {
            StopCoroutine(attackCoroutine);
            attackCoroutine = null;
        }

        if (parryCoroutine != null)
        {
            StopCoroutine(parryCoroutine);
            parryCoroutine = null;
        }

        if (health != null)
            health.SetParrying(false);

        canAttack = true;
        canParry = true;
        CurrentAction = ActionState.Idle;
    }
}
