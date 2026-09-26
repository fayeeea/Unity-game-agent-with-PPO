
using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(CharacterController))]
[RequireComponent(typeof(Animator))]
public class Move : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 5f;
    public float gravity = -9.81f;

    [Header("Camera")]
    public Transform cameraPivot;
    public float mouseSensitivity = 0.1f;
    public float maxLookAngle = 85f;

    private Animator animator;
    private CharacterController controller;

    private float verticalVelocity;
    private float xRotation;

    private Combat combat;

    // EnemyState가 0.05초 단위로 읽는 플레이어 직전 행동.
    public int PreviousMovement { get; private set; }
    public float PreviousRotation { get; private set; }
    public int PreviousCombat { get; private set; }

    private int pendingMovement;
    private float pendingRotation;
    private int pendingCombat;

    public void CapturePreviousAction()
    {
        PreviousMovement = pendingMovement;
        PreviousRotation = Mathf.Clamp(pendingRotation, -30f, 30f);
        PreviousCombat = pendingCombat;

        pendingMovement = 0;
        pendingRotation = 0f;
        pendingCombat = 0;
    }

    public void ResetPreviousAction()
    {
        pendingMovement = 0;
        pendingRotation = 0f;
        pendingCombat = 0;
        PreviousMovement = 0;
        PreviousRotation = 0f;
        PreviousCombat = 0;
        verticalVelocity = 0f;
        xRotation = 0f;
        if (cameraPivot != null)
            cameraPivot.localRotation = Quaternion.identity;
        if (animator != null)
            animator.SetInteger("Move", 0);
    }

    void Start()
    {
        animator = GetComponent<Animator>();
        controller = GetComponent<CharacterController>();
        combat = GetComponent<Combat>();

        // 애니메이션이 캐릭터 위치를 직접 움직이지 않도록 설정
        animator.applyRootMotion = false;

        // 게임 시작 시 마우스 커서 잠금
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        if (cameraPivot != null)
        {
            xRotation = cameraPivot.localEulerAngles.x;

            if (xRotation > 180f)
                xRotation -= 360f;
        }
    }

    void Update()
    {
        Keyboard keyboard = Keyboard.current;
        Mouse mouse = Mouse.current;

        if (keyboard == null)
            return;

        // ESC: 마우스 잠금 해제
        if (keyboard.escapeKey.wasPressedThisFrame)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        // 잠금 해제 상태에서 왼쪽 클릭하면 다시 잠금
        if (mouse != null &&
            mouse.leftButton.wasPressedThisFrame &&
            Cursor.lockState != CursorLockMode.Locked)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        // FPS 카메라 회전
        if (mouse != null &&
            Cursor.lockState == CursorLockMode.Locked)
        {
            Vector2 mouseDelta = mouse.delta.ReadValue();

            float mouseX = mouseDelta.x * mouseSensitivity;
            float mouseY = mouseDelta.y * mouseSensitivity;
            pendingRotation += mouseX;

            // 좌우: 캐릭터 전체 회전
            transform.Rotate(Vector3.up * mouseX);

            // 상하: 카메라만 회전
            xRotation -= mouseY;
            xRotation = Mathf.Clamp(
                xRotation,
                -maxLookAngle,
                maxLookAngle
            );

            if (cameraPivot != null)
            {
                cameraPivot.localRotation =
                    Quaternion.Euler(xRotation, 0f, 0f);
            }
        }

        // WASD 입력
        float horizontal = 0f;
        float vertical = 0f;

        if (keyboard.wKey.isPressed)
            vertical += 1f;

        if (keyboard.sKey.isPressed)
            vertical -= 1f;

        if (keyboard.aKey.isPressed)
            horizontal -= 1f;

        if (keyboard.dKey.isPressed)
            horizontal += 1f;

        // 캐릭터가 바라보는 방향 기준 이동
        Vector3 move =
            transform.right * horizontal +
            transform.forward * vertical;

        // 대각선 이동 속도 보정
        move = Vector3.ClampMagnitude(move, 1f);

        // 중력 처리
        if (controller.isGrounded && verticalVelocity < 0f)
        {
            verticalVelocity = -2f;
        }

        verticalVelocity += gravity * Time.deltaTime;

        Vector3 velocity = move * moveSpeed;
        velocity.y = verticalVelocity;

        controller.Move(velocity * Time.deltaTime);

        // 하체 이동 애니메이션
        int moveState = 0;

        if (vertical != 0f)
        {
            moveState = 1;
        }
        else if (horizontal < 0f)
        {
            moveState = 2;
        }
        else if (horizontal > 0f)
        {
            moveState = 3;
        }

        animator.SetInteger("Move", moveState);
        // 5방향 행동으로 기록: 전후 우선, 이후 좌우.
        if (vertical > 0f) pendingMovement = 1;
        else if (vertical < 0f) pendingMovement = 3;
        else if (horizontal < 0f) pendingMovement = 2;
        else if (horizontal > 0f) pendingMovement = 4;
        else pendingMovement = 0;

        // 공격 / 패링 애니메이션
        // 공격 / 패링 애니메이션
        if (mouse != null)
        {
            if (mouse.leftButton.wasPressedThisFrame)
            {
                animator.SetTrigger("Attack");
                combat.Attack();
                pendingCombat = 1;
            }

            if (mouse.rightButton.wasPressedThisFrame)
            {
                animator.SetTrigger("Defense");
                combat.Parry();
                pendingCombat = 2;
            }
        }
    }
}