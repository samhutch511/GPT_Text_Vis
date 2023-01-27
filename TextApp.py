"""
TextApp.py by Sam Hutchinson, 1/25/2023
"""

import sys
import time
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QApplication, QPushButton, QGridLayout, QWidget, QLabel, QTextEdit, QScrollArea, QVBoxLayout

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from math import *
import colorsys
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

class MainWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        
        #Define window properties
        self.setWindowTitle("Text Analysis")
        self.setGeometry(0, 50, 1450, 750)
        self.buttons = []
        
        #Define ScrollArea and ScrollWidget to add buttons on later
        self.buttonArea = QScrollArea()
        self.buttonArea.setFixedSize(750, 700)
        self.buttonArea.setWidgetResizable(True)
        self.scrollWidget = QWidget()
        self.scrollLayout = QGridLayout()
        self.scrollLayout.setVerticalSpacing(5)
        self.scrollLayout.setHorizontalSpacing(0)
        
        #Define text input boxes
        self.inputText = QTextEdit()
        self.inputText.setPlaceholderText("Enter your text")
        self.inputText.setFixedSize(550, 40)
        self.howMany = QTextEdit()
        self.howMany.setPlaceholderText("Add how many tokens?")
        self.howMany.setFixedSize(550, 25)
        
        #Define button that sets it all off
        self.getText = QPushButton("Process Text")
        self.getText.clicked.connect(self.displayText)
        
        #Define summary label
        self.label = QTextEdit()
        self.label.setReadOnly(True)
        self.label.setFixedSize(550, 40)
        
        #Define place for figure
        self.graph = QLabel()
        self.graph.setFixedSize(550, 550)
        self.graph.setScaledContents(True)
        
        #Add widgets to layout
        self.main_layout = QGridLayout()
        self.main_layout.addWidget(self.inputText, 0, 0, 1, 1)
        self.main_layout.addWidget(self.howMany, 1, 0, 1, 1)
        self.main_layout.addWidget(self.getText, 2, 0, 1, 1)
        self.main_layout.addWidget(self.label, 3, 0, 1, 1)
        self.main_layout.addWidget(self.graph, 4, 0, 1, 1)
        self.main_layout.addWidget(self.buttonArea, 0, 1, 5, 1)
        self.setLayout(self.main_layout)
        
        #Define modeling tools
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        self.tokens = []
        self.probs = []
        self.surprisals = []
            
    
    def addButtons(self, num, labels=None):
        
        #Remove old buttons
        if len(self.buttons) != 0:
            for b in self.buttons:
                self.scrollLayout.removeWidget(b)
        self.buttons = []
        
        #Add new buttons
        if labels == None:
            labels = list(range(num))
            
        for i in range(num):
            but = QLabel(labels[i])
            but.setAlignment(Qt.AlignmentFlag.AlignCenter)
            but.setMaximumHeight(20)
            but.setMinimumHeight(20)
            but.setMaximumWidth(140)
            but.setMinimumWidth(140)
            
            if i != 0:
                #Normalize surprisals and attach to colormap, color buttons accordingly
                norm = matplotlib.colors.Normalize(vmin=min(self.surprisals), vmax=max(self.surprisals))
                cmap = matplotlib.cm.get_cmap('Wistia')
                color = QColor(int(255*cmap(norm(self.surprisals)[i - 1])[0]), int(255*cmap(norm(self.surprisals)[i - 1])[1]), int(255*cmap(norm(self.surprisals)[i - 1])[2]), int(255*cmap(norm(self.surprisals)[i - 1])[3]))
                but.setStyleSheet("background-color: {}".format(color.name()))
            
            but.enterEvent = self.plot
            but.setObjectName(str(i))
            self.buttons.append(but)
            
        for i in range(len(self.buttons)):
            x = (i // 5)
            y = (i % 5) + 10
            self.scrollLayout.addWidget(self.buttons[i], x, y)
            
        if i < 137:
            for j in range(i + 1, 137):
                blank = QLabel()
                blank.setMaximumHeight(20)
                blank.setMinimumHeight(20)
                blank.setMaximumWidth(140)
                blank.setMinimumWidth(140)
                x = (j // 5)
                y = (j % 5) + 10
                self.scrollLayout.addWidget(blank, x, y)
            
        #Apply changes to buttons
        self.scrollWidget.setLayout(self.scrollLayout)
        self.buttonArea.setWidget(self.scrollWidget)

    def plot(self, event):
        for b in self.buttons:
            if b.underMouse():
                num = int(b.objectName())
                color = b.palette().button().color()
                b.setStyleSheet(f"border: 2px solid black; background-color: {color.name()};")
            else:
                color = b.palette().button().color()
                b.setStyleSheet("background-color:" + color.name() +";")
        
        #Display summary stats
        if num != 0:
            self.label.setText(f"Surprisal: {self.surprisals[num - 1]}\nPrediction Entropy: {stats.entropy(self.probs[num])}")
        else:
            self.label.setText(f"Surprisal: NA\nPrediction Entropy: {stats.entropy(self.probs[num])}")
        
        #Display next predictions
        probs = self.probs[num]
        dic = {}
        for i in range(len(probs)):
            dic[probs[i]] = self.tokenizer.decode(i)
        items = list(dic.items())
        items.sort(reverse=True)
        top = items[0:10]
        top_dict = dict(top)
        
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.bar(list(top_dict.values()), list(top_dict.keys()), 0.75)
        plt.xticks(rotation=45, ha="right")
        plt.xticks(fontsize=10)
        xlabel = "Top 10 Most Probable Next Tokens"
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability')
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_title(f"Predicted Token After \'{self.tokens[0][num]}\':")
        plt.savefig('./graphs/fig.jpg')
        plt.close(fig)
        
        pixmap = QPixmap('./graphs/fig.jpg')
        self.graph.setPixmap(pixmap)
        
    def displayText(self):
        
        #Process text
        num = int(self.howMany.toPlainText())
        text = self.inputText.toPlainText()
        
        if num == 0:
            self.tokens = self.get_tokens(text)
            self.probs = self.get_probs(self.tokens[1])
            self.surprisals = self.get_surprisals(self.probs, self.tokens[1])
        else:
            for i in range(num + 1):
                self.tokens = self.get_tokens(text)
                self.probs = self.get_probs(self.tokens[1])
                self.surprisals = self.get_surprisals(self.probs, self.tokens[1])
                
                dist = self.probs[-1]
                choose=np.random.choice([i for i in range(len(dist))], 1, p=dist)
                next_token = str(self.tokenizer.decode(choose[0]))
                text = text + next_token
                
        self.addButtons(num=len(self.tokens[0]), labels=self.tokens[0])

        
    def labelHover(self):
        self.label.setText('Test')
        
    def get_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        tokenized = [self.tokenizer.decode(t) for t in tokens]
        return (tokenized, tokens)

    def get_probs(self, tokens):

        #convert to tensor variable
        tokens_tensor = torch.tensor([tokens])

        #get predictions
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
        predictions = outputs[0]

        #compile probability distribution outputs
        probs_list = [torch.softmax(predictions[0][i],-1).data.numpy() for i in range(len(predictions[0]))]

        return probs_list

    def get_surprisals(self, probs, tokens):
        surprisals = []
        for i in range(len(tokens) - 1):
            token = tokens[i + 1]
            prob = probs[i][token]
            surprisals.append(-1 * log(prob, 10))
        return surprisals
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
