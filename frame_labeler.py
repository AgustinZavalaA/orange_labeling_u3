import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class frame_labeler(QtWidgets.QMainWindow):
    def __init__(self, frame_title: str) -> None:
        super().__init__()
        self.setGeometry(300, 300, 300, 130)
        self.setWindowTitle(frame_title)
        self.initGui()

    def initGui(self) -> None:
        font = QtGui.QFont("Arial", 14)
        self.setFont(font)

        self.size_label = QtWidgets.QLabel("Size:", self)
        self.size_label.move(10, 10)
        self.size_buttons = QtWidgets.QButtonGroup(self)

        self.size_buttons.addButton(QtWidgets.QRadioButton("Small", self), 0)
        self.size_buttons.addButton(QtWidgets.QRadioButton("Medium", self), 1)
        self.size_buttons.addButton(QtWidgets.QRadioButton("Large", self), 2)
        self.size_buttons.button(0).setChecked(True)
        self.size_buttons.button(0).move(20, 40)
        self.size_buttons.button(1).move(20, 60)
        self.size_buttons.button(2).move(20, 80)

        self.state_label = QtWidgets.QLabel("State:", self)
        self.state_label.move(150, 10)
        self.state_buttons = QtWidgets.QButtonGroup(self)

        self.state_buttons.addButton(QtWidgets.QRadioButton("Ripe", self), 0)
        self.state_buttons.addButton(QtWidgets.QRadioButton("Unripe", self), 1)
        self.state_buttons.button(0).setChecked(True)
        self.state_buttons.button(0).move(160, 40)
        self.state_buttons.button(1).move(160, 60)

        # test
        # self.button = QtWidgets.QPushButton("Test", self)
        # self.button.move(10, 100)
        # self.button.clicked.connect(self.print_size_label)

        self.show()

    def print_size_label(self) -> None:
        print(self.size_buttons.checkedButton().text())
        print(self.state_buttons.checkedButton().text())

    def get_label(self) -> str:
        return f"{self.size_buttons.checkedButton().text()},{self.state_buttons.checkedButton().text()}"


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = frame_labeler("Frame Labeler")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
