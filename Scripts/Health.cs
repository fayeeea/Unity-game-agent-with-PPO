using System;
using System.Collections;
using UnityEngine;

public class Health : MonoBehaviour
{
    public enum DamageResult
    {
        Ignored,   // 이미 죽었거나 유효하지 않은 피해
        Parried,   // 패링으로 피해 차단
        Damaged    // 실제 체력 감소
    }

    [Header("Health")]
    public int maxHealth = 100;
    public int currentHealth;

    [Header("Hit Effect")]
    public Color hitColor = Color.red;
    public float hitDuration = 0.15f;

    [Header("Respawn")]
    public float respawnDelay = 2f;
    public Vector3 respawnPosition = Vector3.zero;

    [Header("PPO Episode")]
    public bool externallyManagedRespawn = false;

    // 기존 이벤트 유지
    public event Action OnDied;
    public event Action OnRespawned;

    // Reward 연결용: 실제 피해량 / 실제 공격을 패링한 횟수
    public event Action<int> OnDamaged;
    public event Action OnParrySuccess;

    private bool isParrying;
    private bool isDead;

    private Renderer[] renderers;
    private MaterialPropertyBlock propertyBlock;
    private Coroutine hitCoroutine;
    private Coroutine respawnCoroutine;

    private CharacterController controller;
    private Move move;
    private Combat combat;
    private Animator animator;

    // EnemyState.Start()의 첫 상태 수집보다 먼저 체력이 초기화되도록 Awake 사용.
    void Awake()
    {
        currentHealth = maxHealth;

        renderers = GetComponentsInChildren<Renderer>();
        propertyBlock = new MaterialPropertyBlock();

        controller = GetComponent<CharacterController>();
        move = GetComponent<Move>();
        combat = GetComponent<Combat>();
        animator = GetComponent<Animator>();
    }

    public void SetParrying(bool value)
    {
        isParrying = !isDead && value;
    }

    // 기존 스크립트의 TakeDamage(...) 호출을 깨지 않도록 유지
    public void TakeDamage(int damage)
    {
        ApplyDamage(damage);
    }

    // Combat에서 피해가 실제로 들어갔는지 확인할 때 사용
    public DamageResult ApplyDamage(int damage)
    {
        if (isDead || currentHealth <= 0 || damage <= 0)
            return DamageResult.Ignored;

        if (isParrying)
        {
            Debug.Log(gameObject.name + " 패링 성공!");
            OnParrySuccess?.Invoke();
            return DamageResult.Parried;
        }

        int healthBefore = currentHealth;
        currentHealth = Mathf.Max(currentHealth - damage, 0);
        int actualDamage = healthBefore - currentHealth;

        // 사망 처리 전 상태를 고정해 중복 사망을 방지
        bool died = currentHealth <= 0;
        if (died)
            isDead = true;

        Debug.Log(gameObject.name + " 체력: " + currentHealth + " / " + maxHealth);
        OnDamaged?.Invoke(actualDamage);

        if (died)
        {
            if (externallyManagedRespawn)
                BeginDeath();
            else
                respawnCoroutine = StartCoroutine(RespawnRoutine());
            return DamageResult.Damaged;
        }

        // 살아 있을 때 피격 색상 효과
        if (hitCoroutine != null)
            StopCoroutine(hitCoroutine);

        hitCoroutine = StartCoroutine(HitFlash());
        return DamageResult.Damaged;
    }

    IEnumerator HitFlash()
    {
        SetCharacterColor(hitColor);

        yield return new WaitForSeconds(hitDuration);

        ClearCharacterColor();
        hitCoroutine = null;
    }

    private void BeginDeath()
    {
        isParrying = false;
        OnDied?.Invoke();
        Debug.Log(gameObject.name + " 사망!");

        if (hitCoroutine != null)
        {
            StopCoroutine(hitCoroutine);
            hitCoroutine = null;
        }
        ClearCharacterColor();

        if (move != null)
            move.enabled = false;
        if (combat != null)
            combat.enabled = false;

        foreach (Renderer renderer in renderers)
        {
            if (renderer != null)
                renderer.enabled = false;
        }
    }

    IEnumerator RespawnRoutine()
    {
        BeginDeath();
        yield return new WaitForSeconds(respawnDelay);
        respawnCoroutine = null;
        ResetForNewEpisode();
    }

    // Python의 episode_end 응답 후 양쪽을 함께 리셋하는 데 사용.
    public void ResetForNewEpisode()
    {
        if (respawnCoroutine != null)
        {
            StopCoroutine(respawnCoroutine);
            respawnCoroutine = null;
        }
        if (hitCoroutine != null)
        {
            StopCoroutine(hitCoroutine);
            hitCoroutine = null;
        }
        ClearCharacterColor();

        if (controller != null)
            controller.enabled = false;

        transform.position = respawnPosition;
        transform.rotation = Quaternion.identity;

        if (controller != null)
            controller.enabled = true;

        currentHealth = maxHealth;
        isParrying = false;

        if (animator != null)
        {
            animator.SetInteger("Move", 0);
            animator.ResetTrigger("Attack");
            animator.ResetTrigger("Defense");
            animator.Rebind();
            animator.Update(0f);
        }

        foreach (Renderer renderer in renderers)
        {
            if (renderer != null)
                renderer.enabled = true;
        }

        isDead = false;
        OnRespawned?.Invoke();

        if (combat != null)
            combat.enabled = true;
        if (move != null)
            move.enabled = true;

        Debug.Log(gameObject.name + " 부활!");
    }

    void SetCharacterColor(Color color)
    {
        foreach (Renderer renderer in renderers)
        {
            if (renderer == null)
                continue;

            for (int i = 0; i < renderer.sharedMaterials.Length; i++)
            {
                renderer.GetPropertyBlock(propertyBlock, i);

                propertyBlock.SetColor("_BaseColor", color);
                propertyBlock.SetColor("_Color", color);

                renderer.SetPropertyBlock(propertyBlock, i);
            }
        }
    }

    void ClearCharacterColor()
    {
        foreach (Renderer renderer in renderers)
        {
            if (renderer == null)
                continue;

            for (int i = 0; i < renderer.sharedMaterials.Length; i++)
            {
                propertyBlock.Clear();
                renderer.SetPropertyBlock(propertyBlock, i);
            }
        }
    }
}
