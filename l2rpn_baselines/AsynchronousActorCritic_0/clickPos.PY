import pyautogui
import time

# 设置点击间隔时间（秒）
click_interval = 60 * 3

try:
    while True:
        # 模拟按下 ALT 键
        pyautogui.keyDown('alt')

        # 等待一段时间
        time.sleep(0.1)  # 可以根据需要调整等待时间

        # 模拟释放 ALT 键
        pyautogui.keyUp('alt')

        # 等待一段时间
        time.sleep(0.1)  # 可以根据需要调整等待时间

        # 模拟按下 ALT 键
        pyautogui.keyDown('alt')

        # 等待一段时间
        time.sleep(0.1)  # 可以根据需要调整等待时间

        # 模拟释放 ALT 键
        pyautogui.keyUp('alt')

        print("\n do clean.")

        # 等待指定的点击间隔时间
        time.sleep(click_interval - 0.4)  # 减去按下和释放 ALT 键的时间

except KeyboardInterrupt:
    print("\nScript stopped.")