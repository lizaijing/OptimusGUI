# main_window.py

import base64
import re
import threading
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap, QTextBlockFormat, QTextCursor
from PyQt6.QtWebSockets import QWebSocket
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor

from server.api import Server


class GUIStatus(Enum):
    INIT = 1
    RESET = 2
    RUNNING = 3


class MainWindow(QMainWindow):
    agent_response_signal = pyqtSignal(str)
    reset_done_signal = pyqtSignal(object, object)

    def __init__(self, server: Server):
        super().__init__()
        self.server = server
        self.gui_status = GUIStatus.INIT
        self.selected_task: Optional[str] = None
        # 图像缓冲
        self.image_buffer = deque(maxlen=100)
        self.original_pixmap: Optional[QPixmap] = None

        self.pause_default_style = "color: #FFFFFF; background-color: #555555;"
        self.pause_active_style = "color: #000000; background-color: #79D5A9;"
        self.reset_default_style = "color: #FFFFFF; background-color: #555555;"
        self.reset_active_style = "color: #000000; background-color: #79D5A9;"
        # 信号连接
        self.agent_response_signal.connect(self.receive_agent_response)
        self.reset_done_signal.connect(self._finish_reset)
        tasks = {
            "Planning": "planning",
            "Action": "action",
            "Captioning": "captioning",
            "Embodied QA": "embodied_qa",
            "Grounding": "grounding",
        }
        # 窗口设置
        self.setWindowTitle("Optimus-3 Agent")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("background-color: #2E2E2E;")

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(main_splitter)

        # 左侧容器：环境画面 + 功能按钮
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        # 左侧：游戏画面
        self.image_label = QLabel("Waiting for server...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.image_label.setScaledContents(True)
        left_layout.addWidget(self.image_label, stretch=1)
        self.btn_default_style = "color: #FFFFFF; background-color: #555555;"
        self.btn_selected_style = "color: #000000; background-color: #FFA92D;"
        self.task_buttons = {}  # task_type -> QPushButton
        btn_layout = QHBoxLayout()
        # btn_names = ["Planning", "Action", "Captioning", "Embodied QA", "Grounding"]
        for label, task in tasks.items():
            btn = QPushButton(label)
            btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            btn.setStyleSheet(self.btn_default_style)
            btn.clicked.connect(lambda _, t=task: self._select_task(t))
            btn_layout.addWidget(btn)
            self.task_buttons[task] = btn
        left_layout.addLayout(btn_layout)
        main_splitter.addWidget(left_widget)

        # 右侧：消息区 + 输入框 + 按钮
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # 消息显示区：QTextEdit，自动换行，无横向滚动
        self.message_area = QTextEdit()
        self.message_area.setReadOnly(True)
        self.message_area.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.message_area.setStyleSheet("background: #3C3C3C; color: #E0E0E0;")
        self.message_area.setFont(QFont("Arial", 12))
        self.default_color = self.message_area.textColor()
        self.system_color = QColor("#79D5A9")  # System 消息：绿色
        self.agent_color = QColor("#6096E6")  # Agent 消息：蓝色
        self.user_color = QColor("#FFA92D")  # User 消息：黄色
        right_splitter.addWidget(self.message_area)

        # 输入框
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Arial", 12))
        self.input_field.setPlaceholderText("Enter your command here...")
        self.input_field.setStyleSheet("background-color: #3C3C3C; color: #FFFFFF;")
        self.input_field.returnPressed.connect(self.handle_input)
        right_splitter.addWidget(self.input_field)

        right_splitter.setSizes([600, 150])
        right_layout.addWidget(right_splitter)

        # 按钮区
        btn_layout = QHBoxLayout()
        self.pause_button = QPushButton("Pause Agent")
        self.pause_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.pause_button.setStyleSheet(self.pause_default_style)
        self.pause_button.clicked.connect(self.handle_pause)
        btn_layout.addWidget(self.pause_button)

        self.reset_button = QPushButton("Reset Environment")
        self.reset_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.reset_button.setStyleSheet(self.reset_default_style)
        self.reset_button.clicked.connect(self.handle_reset_environment)
        btn_layout.addWidget(self.reset_button)

        right_layout.addLayout(btn_layout)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([1100, 300])

        self.statusBar().showMessage("Initializing...")

        # 定时刷新游戏画面
        self.image_display_timer = QTimer(self)
        self.image_display_timer.timeout.connect(self._update_image_from_buffer)
        self.image_display_timer.start(50)

        # 启动 WebSocket
        self.gui_status = GUIStatus.RUNNING
        self.start_websocket_listener()

        # 显示初始文本
        try:
            init_data = self.server.get_initial_text()
            text = init_data.get("text") if init_data else None
        except:
            text = None
        if not text:
            text = "hello, I'm Optimus-3."

        self._typewriter(text, prefix="Agent: ")
        # self.statusBar().showMessage("Received initial text from server.")

    def _select_task(self, task: str):
        """记录用户选中的任务类型，并在消息区提示。"""
        self.selected_task = task
        for t, btn in self.task_buttons.items():
            if t == task:
                btn.setStyleSheet(self.btn_selected_style)
            else:
                btn.setStyleSheet(self.btn_default_style)
        self.message_area.setTextColor(self.system_color)
        self.message_area.append(f"System: Selected task → {task}\n\n")
        if self.selected_task == "action":
            self.handle_input()

    def handle_input(self):
        if self.selected_task == "action":
            threading.Thread(target=self._send_command_to_server, args=(None, self.selected_task), daemon=True).start()
            return
        cmd = self.input_field.text().strip()
        if not cmd:
            return
        if self.selected_task is None:
            self.message_area.setTextColor(self.system_color)
            self.message_area.append("System: Please select a task before sending.\n\n")
            return
        # 1) 立刻在消息区插入一个“右对齐”段落
        cursor = self.message_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignRight)
        cursor.insertBlock(block_fmt)
        cursor_char_fmt = cursor.charFormat()
        cursor_char_fmt.setForeground(self.user_color)
        cursor.insertText(f"You: {cmd}", cursor_char_fmt)
        self.message_area.ensureCursorVisible()

        cursor.insertBlock()
        self.message_area.ensureCursorVisible()
        self.input_field.clear()

        # 2) 异步发送给服务端
        threading.Thread(target=self._send_command_to_server, args=(cmd, self.selected_task), daemon=True).start()

    def _send_command_to_server(self, cmd: str | None, task: str):
        print(cmd, task)
        try:
            if task == "action":
                while self.gui_status == GUIStatus.RUNNING:
                    self.server.send_text("action", task)
                    print("action", self.gui_status)
                resp = "Action"
            else:
                data = self.server.send_text(cmd, task)
                resp = data.get("response", "No response.") if data else "No response."
        except Exception as e:
            resp = f"Error: {e}"
        self.agent_response_signal.emit(resp + "|" + task)

    def receive_agent_response(self, args):
        # Agent 回复使用打字机形式，左对齐
        resp, cmd = args.split("|")[0], args.split("|")[1]
        self._typewriter(resp, prefix="Agent: ")
        self.statusBar().showMessage("Agent responded. Ready.")

        if cmd == "grounding":
            # TODO: draw a bounding box on the image

            # 2. 创建 QPainter 并开始在 QPixmap 上绘制
            painter = QPainter(self.original_pixmap)

            # 3. 创建一个红色的 QPen
            pen = QPen(QColor("red"))
            pen.setWidth(3)  # 设置线条宽度为 3 像素

            # 4. 应用画笔
            painter.setPen(pen)

            # 5. 使用 QRect 或两个坐标点绘制矩形
            # Qt 的 drawRect 直接使用左上角坐标和宽高，
            # 所以我们需要根据右下角坐标计算出宽度和高度。
            numbers = re.findall(r"\d+", resp)
            coords = [int(num) for num in numbers][1:5]
            x1, y1, x2, y2 = coords
            top_left = QPoint(x1, y1)
            bottom_right = QPoint(x2, y2)
            rect = QRect(top_left, bottom_right)
            painter.drawRect(rect)

            # 6. 结束绘制，释放资源
            painter.end()
            self._display_pixmap(self.original_pixmap)

    def _append_agent_text(self, text: str):
        # 以普通文本左对齐插入一段
        cursor = self.message_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        cursor.insertBlock(block_fmt)
        fmt = cursor.charFormat()
        fmt.setForeground(self.agent_color)
        cursor.insertText(f"Agent: {text}", fmt)
        self.message_area.ensureCursorVisible()

    def _typewriter(self, text: str, prefix: str = "Agent: "):
        # 打字机效果
        cursor = self.message_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        cursor.insertBlock(block_fmt)
        self.message_area.setTextCursor(cursor)
        self.message_area.setTextColor(self.agent_color)
        self.message_area.insertPlainText(prefix)
        self.message_area.ensureCursorVisible()

        def append_char(idx=0):
            if idx < len(text):
                self.message_area.insertPlainText(text[idx])
                self.message_area.ensureCursorVisible()
                QTimer.singleShot(10, lambda: append_char(idx + 1))
            else:
                self.message_area.insertPlainText("\n\n")
                self.message_area.ensureCursorVisible()

        append_char()

    def resizeEvent(self, event):
        # 窗口大小变动时刷新画面
        if self.original_pixmap and not self.original_pixmap.isNull():
            self._display_pixmap(self.original_pixmap)
        super().resizeEvent(event)

    def handle_pause(self):
        """
        点击“Pause Agent”后，发送 'pause' 或 'resume' 给服务端，
        服务端收到后停止/恢复 environment 更新，客户端不再收到新帧。
        """
        if self.gui_status == GUIStatus.RUNNING:
            self.gui_status = GUIStatus.INIT  # 标记为暂停状态
            self.pause_button.setText("Resume Agent")
            self.pause_button.setStyleSheet(self.pause_active_style)
            self.reset_button.setStyleSheet(self.reset_default_style)
            self.message_area.setTextColor(self.system_color)
            self.message_area.append("System: Agent paused.\n\n")
            self.statusBar().showMessage("Agent Paused")
            self.server.pause()
        else:
            self.gui_status = GUIStatus.RUNNING
            self.pause_button.setText("Pause Agent")
            self.pause_button.setStyleSheet(self.pause_default_style)
            self.message_area.setTextColor(self.system_color)
            self.message_area.append("System: Agent resumed.\n\n")
            self.statusBar().showMessage("Agent Resumed")
            self.server.resume()
            if self.selected_task == "action":
                threading.Thread(
                    target=self._send_command_to_server, args=(None, self.selected_task), daemon=True
                ).start()
            else:
                self.task_buttons[self.selected_task].setStyleSheet(self.btn_default_style)

    def handle_reset_environment(self):
        """
        点击“Reset Environment”后：
         - 清空缓冲区与当前显示
         - 向服务端发送 reset 请求，获得初始图像并加入缓冲区
        """
        self.reset_button.setStyleSheet(self.reset_active_style)
        self.pause_button.setStyleSheet(self.pause_default_style)
        self.message_area.setTextColor(self.system_color)
        self.message_area.append("System: environment reset...\n\n")
        self.statusBar().showMessage("Resetting environment...")
        self.image_buffer.clear()
        self.original_pixmap = None
        self._display_pixmap(None)
        self.image_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.image_label.setStyleSheet("color: #FFFFFF;")
        self.image_label.setText("Resetting... Waiting for new observation...")
        threading.Thread(target=self._run_reset, daemon=True).start()

    def _run_reset(self):
        """
        后台线程：调用服务端 reset，并在完成后切回主线程处理结果。
        """
        try:
            data = self.server.reset()
            self.reset_done_signal.emit(data, None)
            # QTimer.singleShot(0, lambda d=data: self._finish_reset(d, None))
        except Exception as e:
            # QTimer.singleShot(0, lambda: self._finish_reset(None, e))
            self.reset_done_signal.emit(None, e)

    def _finish_reset(self, data, error):
        """
        主线程：处理 reset 返回或异常，然后恢复 Reset 按钮样式。
        """
        if error:
            self.message_area.setTextColor(self.system_color)
            self.message_area.append(f"System: Reset failed: {error}")
        else:
            if data and data.get("observation"):
                pix = self._base64_to_pixmap(data["observation"])
                if pix:
                    self.image_buffer.append(pix)
                    self.original_pixmap = pix
                    self.statusBar().showMessage("Environment reset. Displaying new observation.")
                    self.gui_status = GUIStatus.RUNNING
            else:
                self.message_area.setTextColor(self.system_color)
                self.message_area.append("System: Reset returned no observation.")

        # 3) 完成后恢复 Reset 按钮默认样式
        self.reset_button.setStyleSheet(self.reset_default_style)

    def start_websocket_listener(self):
        self.ws = QWebSocket()
        parsed = QUrl(self.server.url)
        host = parsed.host()
        ws_url = QUrl(f"ws://{host}:{self.server.port}/ws/obs")
        self.ws.errorOccurred.connect(lambda e: print(f"WebSocket error: {e}"))
        self.ws.connected.connect(self._on_ws_connected)
        self.ws.disconnected.connect(lambda: print("WebSocket disconnected"))
        self.ws.textMessageReceived.connect(self._on_ws_message)
        print(f"Connecting WebSocket to {ws_url.toString()}...")
        self.ws.open(ws_url)

    def _on_ws_connected(self):
        print("WebSocket connected")
        # 初次连接，通过 REST 获取当前 obs
        try:
            b64 = self.server.receive_obs()
            if b64:
                pix = self._base64_to_pixmap(b64)
                if pix:
                    self.image_buffer.append(pix)
                    self.original_pixmap = pix
        except Exception as e:
            self.message_area.addItem(QListWidgetItem(f"System: Failed to get initial observation via REST: {e}"))

    def _on_ws_message(self, b64_str: str):
        if not b64_str:
            return
        img_data = base64.b64decode(b64_str)
        qimg = QImage.fromData(img_data)
        if qimg.isNull():
            return
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            640, 360, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_buffer.append(pixmap)
        self.original_pixmap = pixmap

    def _update_image_from_buffer(self):
        if self.image_buffer:
            pix = self.image_buffer.popleft()
        else:
            pix = self.original_pixmap
        self._process_and_display_pixmap(pix)

    def _process_and_display_pixmap(self, pixmap: QPixmap):
        if pixmap and not pixmap.isNull():
            self.original_pixmap = pixmap
            self._display_pixmap(pixmap)
        # else:
        #     self.message_area.addItem(QListWidgetItem("Error: Could not load image for display."))

    def _display_pixmap(self, pix: Optional[QPixmap]):
        if pix and not pix.isNull():
            self.image_label.setPixmap(
                pix.scaled(
                    640,
                    360,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            if self.original_pixmap:
                self.image_label.setPixmap(
                    self.original_pixmap.scaled(
                        640,
                        360,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                self.image_label.setText("No Image")

    def _base64_to_pixmap(self, b64_str: str) -> Optional[QPixmap]:
        try:
            data = base64.b64decode(b64_str)
            qimg = QImage.fromData(data)
            if qimg.isNull():
                return None
            pix = QPixmap.fromImage(qimg)
            return pix.scaled(
                640, 360, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        except Exception:
            return None
