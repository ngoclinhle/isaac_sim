import os
import sys
import pickle

from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QPushButton, QFileDialog, QVBoxLayout, QWidget
import numpy as np


class DataViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Data browser")
        self.setGeometry(100, 100, 600, 400)

        # Set up the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Set up the tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(
            ['KeyName', 'ValueType', 'Size', 'Value'])

        # add a button to browse the filesystem for pickle files
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_on_click)

        self.layout.addWidget(self.tree_widget)
        self.layout.addWidget(self.load_button)

    def set_tree_item_text(self, item: QTreeWidgetItem, key, value):
        item.setText(0, f'{key}')
        item.setText(1, f'{type(value)}')
        if isinstance(value, dict) or isinstance(value, list) or isinstance(value, tuple):
            item.setText(2, f'{len(value)}')
        elif isinstance(value, np.ndarray):
            item.setText(2, f'{value.shape}')
        if hasattr(value, '__str__'):
            item.setText(3, str(value)[:100])

    def load_on_click(self, data):
        current_dir = os.getcwd()
        file = QFileDialog.getOpenFileName(self, 'Open file', current_dir)[0]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.tree_widget.clear()
        self.set_data(data)

    def set_data(self, data_dict: dict, parent: QTreeWidgetItem = None):
        if parent is None:
            root = QTreeWidgetItem(
                self.tree_widget, ['root', str(type(data_dict))])
        else:
            root = parent

        for key in data_dict:
            child = QTreeWidgetItem(root)
            self.set_tree_item_text(child, key, data_dict[key])
            if isinstance(data_dict[key], dict):
                self.set_data(data_dict[key], child)


app = QApplication(sys.argv)
window = DataViewer()
window.show()
app.exec()
