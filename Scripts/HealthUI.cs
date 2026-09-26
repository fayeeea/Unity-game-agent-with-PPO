
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HealthUI : MonoBehaviour
{
    [Header("Player Health")]
    public Health playerHealth;

    [Header("UI")]
    public Slider healthBar;
    public TMP_Text healthText;

    void Update()
    {
        if (playerHealth == null)
            return;

        // 체력바 업데이트
        if (healthBar != null)
        {
            healthBar.maxValue = playerHealth.maxHealth;
            healthBar.value = playerHealth.currentHealth;
        }

        // 체력 숫자 업데이트
        if (healthText != null)
        {
            healthText.text =
                playerHealth.currentHealth +
                " / " +
                playerHealth.maxHealth;
        }
    }
}