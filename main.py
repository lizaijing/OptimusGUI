import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window_test import MainWindow
from server.api import Server


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # input the url, e.g., http://10.xx.xx.xxx
    server = Server(url="http://10.xx.xx.xx", port=9500)

    gui = MainWindow(server)

    gui.show()
    # gui.receive_image_from_server()

    sys.exit(app.exec())
