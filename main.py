import sys
import os
import numpy as np
import cv2
import imagehash
from PIL import Image, ImageDraw, ImageFont

# --- PyQt5 라이브러리 ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QGroupBox, QListWidget, QMessageBox, QFileDialog,
                             QSplitter, QFrame, QStackedWidget, QScrollArea, 
                             QFormLayout, QAbstractItemView, QListWidgetItem,
                             QComboBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QRect, QRectF, QSize, QPoint, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QPainterPath, QRegion, QFont

from skimage.metrics import structural_similarity as compare_ssim

# --- 설정 ---
OUTPUT_FOLDER = "captured_scores"

# --- 프로페셔널 스타일시트 ---
MODERN_STYLESHEET = """
/* 메인 윈도우 */
QMainWindow {
    background-color: #f5f5f5;
}

QWidget {
    font-family: 'Segoe UI', 'Apple SD Gothic Neo', sans-serif;
    font-size: 12px;
    color: #333333;
}

/* 그룹박스 */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    margin-top: 6px;
    padding-top: 14px;
    font-weight: 600;
    font-size: 11px;
    color: #2c2c2c;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 3px 10px;
    color: #2c2c2c;
}

/* 입력 필드 */
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 6px 10px;
    color: #333333;
    font-size: 12px;
}

QLineEdit:focus {
    border: 1px solid #0078d4;
    background-color: #ffffff;
}

QLineEdit::placeholder {
    color: #999999;
}

/* 기본 버튼 */
QPushButton {
    background-color: #0078d4;
    border: none;
    border-radius: 4px;
    padding: 8px 14px;
    color: #ffffff;
    font-weight: 600;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #106ebe;
}

QPushButton:pressed {
    background-color: #005a9e;
}

QPushButton:disabled {
    background-color: #e0e0e0;
    color: #a0a0a0;
}

/* 영역 선택 버튼 */
QPushButton#selectButton {
    background-color: #6a5acd;
    color: #ffffff;
}

QPushButton#selectButton:hover {
    background-color: #5b4ab8;
}

QPushButton#selectButton:pressed {
    background-color: #4c3ba3;
}

/* 캡처 시작 버튼 */
QPushButton#captureButton {
    background-color: #28a745;
    color: #ffffff;
    font-size: 13px;
    padding: 10px 14px;
    font-weight: 700;
}

QPushButton#captureButton:hover {
    background-color: #218838;
}

QPushButton#captureButton:pressed {
    background-color: #1e7e34;
}

/* 캡처 중지 버튼 (활성화 시) */
QPushButton#captureButtonActive {
    background-color: #dc3545;
    color: #ffffff;
    font-size: 13px;
    padding: 10px 14px;
    font-weight: 700;
}

QPushButton#captureButtonActive:hover {
    background-color: #c82333;
}

QPushButton#captureButtonActive:pressed {
    background-color: #bd2130;
}

/* PDF 생성 버튼 */
QPushButton#pdfButton {
    background-color: #ff8c00;
    color: #ffffff;
    font-size: 13px;
    padding: 10px 14px;
    font-weight: 700;
}

QPushButton#pdfButton:hover {
    background-color: #e67e00;
}

QPushButton#pdfButton:pressed {
    background-color: #cc7000;
}

/* 삭제 버튼 */
QPushButton#deleteButton {
    background-color: #6c757d;
    color: #ffffff;
    padding: 6px 12px;
}

QPushButton#deleteButton:hover {
    background-color: #5a6268;
}

QPushButton#deleteButton:pressed {
    background-color: #545b62;
}

/* 리스트 위젯 */
QListWidget {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 3px;
    color: #333333;
    outline: none;
}

QListWidget::item {
    background-color: #fafafa;
    border-radius: 3px;
    padding: 6px 8px;
    margin: 2px;
    border: 1px solid #e8e8e8;
    color: #333333;
}

QListWidget::item:selected {
    background-color: #0078d4;
    color: #ffffff;
    border: 1px solid #0078d4;
}

QListWidget::item:hover {
    background-color: #e8f4fd;
    border: 1px solid #0078d4;
    color: #333333;
}

QListWidget::item:selected:hover {
    background-color: #106ebe;
    color: #ffffff;
}

/* 라벨 */
QLabel#statusLabel {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 8px;
    font-weight: 600;
    color: #333333;
    font-size: 11px;
}

QLabel#previewLabel {
    background-color: #fafafa;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 6px;
}

QLabel#headerLabel {
    font-size: 16px;
    font-weight: 700;
    color: #2c2c2c;
    padding: 6px 0px;
}

QLabel#sectionLabel {
    font-size: 11px;
    font-weight: 600;
    color: #666666;
    padding: 4px 0px;
}

/* 스플리터 */
QSplitter::handle {
    background-color: #d0d0d0;
    width: 1px;
}

QSplitter::handle:hover {
    background-color: #0078d4;
}

/* 스크롤바 */
QScrollBar:vertical {
    background: #f5f5f5;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #c0c0c0;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #a0a0a0;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 프레임 */
QFrame#leftPanel {
    background-color: #f5f5f5;
    border-right: 1px solid #d0d0d0;
}
"""

class SelectionOverlay(QWidget):
    """영역 선택 오버레이"""
    selection_finished = pyqtSignal(dict) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.start_pos = None
        self.current_pos = None
        self.is_selecting = False
        self.mode_active = False
        self.is_locked = False
        self.confirmed_rect = None
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.hide()

    def set_active(self, active):
        self.mode_active = active
        self.is_locked = False
        if active:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False) 
            self.setCursor(Qt.CrossCursor)
            self.show()
        else:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_lock(self, lock):
        self.is_locked = lock
        self.mode_active = False
        if lock:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.setCursor(Qt.ForbiddenCursor)
            self.show()
        else:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.setCursor(Qt.ArrowCursor)
            self.hide()
        self.update()

    def mousePressEvent(self, event):
        if self.mode_active and event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.current_pos = event.pos()
            self.is_selecting = True
            self.confirmed_rect = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.mode_active and self.is_selecting:
            self.current_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode_active and event.button() == Qt.LeftButton:
            self.is_selecting = False
            rect = QRect(self.start_pos, self.current_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                global_top_left = self.mapToGlobal(rect.topLeft())
                final_area = {
                    'top': global_top_left.y(),
                    'left': global_top_left.x(),
                    'width': rect.width(),
                    'height': rect.height()
                }
                self.confirmed_rect = rect
                self.selection_finished.emit(final_area)
                self.set_active(False)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = None
        if self.is_selecting and self.start_pos and self.current_pos:
            rect = QRect(self.start_pos, self.current_pos).normalized()
        elif self.confirmed_rect:
            rect = self.confirmed_rect

        if self.mode_active or self.is_locked:
            color_alpha = 160 if self.mode_active else 60
            overlay_color = QColor(0, 0, 0, color_alpha)
            path = QPainterPath()
            path.addRect(QRectF(self.rect()))
            if rect:
                path.addRect(QRectF(rect))
            painter.fillPath(path, overlay_color)

        if rect:
            if self.is_locked:
                pen = QPen(QColor(220, 53, 69), 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(rect)
                
                bg_rect = QRect(rect.left(), rect.top() - 28, rect.width(), 28)
                painter.fillRect(bg_rect, QColor(220, 53, 69, 230))
                
                painter.setPen(Qt.white)
                font = QFont("Segoe UI", 9, QFont.Bold)
                painter.setFont(font)
                painter.drawText(bg_rect, Qt.AlignCenter, "캡처 진행 중 (조작 금지)")
                
            elif self.mode_active:
                pen = QPen(QColor(0, 120, 212), 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(rect)
                
                bg_rect = QRect(rect.left(), rect.top() - 26, 180, 26)
                painter.fillRect(bg_rect, QColor(0, 120, 212, 230))
                
                painter.setPen(Qt.white)
                font = QFont("Segoe UI", 9, QFont.Bold)
                painter.setFont(font)
                text = f"{rect.width()} × {rect.height()} px"
                painter.drawText(bg_rect, Qt.AlignCenter, text)
                
            else:
                pen = QPen(QColor(40, 167, 69), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(rect)
                
        elif self.mode_active:
            painter.setPen(Qt.white)
            font = QFont("Segoe UI", 13, QFont.Normal)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "드래그하여 악보 영역을 선택하세요")

class ScoreEditorWidget(QWidget):
    """PDF 생성 전 메타데이터 입력 및 미리보기 위젯"""
    save_requested = pyqtSignal(dict)
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_files = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 상단 설정 영역 (가로 배치)
        top_layout = QHBoxLayout()

        # 1. 메타데이터 입력
        info_group = QGroupBox("악보 정보 입력")
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("노래 제목을 입력하세요")
        self.composer_edit = QLineEdit()
        self.composer_edit.setPlaceholderText("작곡가/아티스트를 입력하세요")
        
        self.title_edit.textChanged.connect(self.refresh_preview)
        self.composer_edit.textChanged.connect(self.refresh_preview)
        
        form_layout.addRow("제목:", self.title_edit)
        form_layout.addRow("작곡가:", self.composer_edit)
        info_group.setLayout(form_layout)
        top_layout.addWidget(info_group)
        
        # 1.5 PDF 설정
        settings_group = QGroupBox("PDF 설정")
        settings_layout = QFormLayout()
        
        self.margin_edit = QLineEdit("60")
        self.spacing_edit = QLineEdit("40")
        self.page_num_pos = QComboBox()
        self.page_num_pos.addItems(["하단 중앙", "하단 우측", "상단 우측", "없음"])
        
        settings_layout.addRow("페이지 여백 (px):", self.margin_edit)
        settings_layout.addRow("이미지 간격 (px):", self.spacing_edit)
        settings_layout.addRow("페이지 번호:", self.page_num_pos)
        settings_group.setLayout(settings_layout)
        top_layout.addWidget(settings_group)
        
        layout.addLayout(top_layout)
        
        self.margin_edit.textChanged.connect(self.refresh_preview)
        self.spacing_edit.textChanged.connect(self.refresh_preview)
        self.page_num_pos.currentIndexChanged.connect(self.refresh_preview)

        # 2. 미리보기 영역
        preview_label = QLabel("페이지 미리보기 (왼쪽 목록에서 순서 변경 가능)")
        preview_label.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(preview_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #e0e0e0; border: 1px solid #ccc;")
        
        self.preview_content = QWidget()
        self.preview_content.setStyleSheet("background-color: #e0e0e0;")
        self.preview_layout = QVBoxLayout(self.preview_content)
        self.preview_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.preview_layout.setSpacing(20)
        
        self.scroll.setWidget(self.preview_content)
        layout.addWidget(self.scroll)

        # 3. 하단 버튼
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("뒤로 가기")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        
        self.btn_save = QPushButton("PDF 저장하기")
        self.btn_save.setObjectName("captureButton") # 초록색 스타일 재사용
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(lambda: self.save_requested.emit({
            'title': self.title_edit.text(),
            'composer': self.composer_edit.text(),
            'margin': self.margin_edit.text(),
            'spacing': self.spacing_edit.text(),
            'page_num_pos': self.page_num_pos.currentText()
        }))
        
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

    def refresh_preview(self):
        if self.current_files:
            self.load_preview(self.current_files)

    def load_preview(self, file_paths):
        self.current_files = file_paths
        # 기존 위젯 제거
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not file_paths:
            return

        try:
            # 설정값 파싱
            try:
                margin = int(self.margin_edit.text())
                spacing = int(self.spacing_edit.text())
            except ValueError:
                margin = 60
                spacing = 40
            page_num_pos_str = self.page_num_pos.currentText()

            # 첫 번째 이미지로 기준 너비 설정 (PDF 생성 로직과 동일하게)
            first_img = Image.open(file_paths[0])
            base_width = first_img.width
            # A4 비율 (210x297mm) -> 높이 계산
            page_height = int(base_width * (297 / 210))
            
            # 화면 표시용 스케일 (미리보기 너비 500px 기준)
            PREVIEW_WIDTH = 500
            scale = PREVIEW_WIDTH / base_width
            
            display_margin = int(margin * scale)
            display_spacing = int(spacing * scale)
            display_content_width = int((base_width - 2*margin) * scale)
            if display_content_width < 10: display_content_width = 10
            
            current_y = margin
            
            # 제목/작곡가 높이 계산
            title = self.title_edit.text().strip()
            composer = self.composer_edit.text().strip()
            
            if title or composer:
                # PDF 생성 로직과 유사하게 높이 추정
                t_h = (base_width/30) * 1.5 if title else 0
                c_h = (base_width/60) * 1.5 if composer else 0
                header_height = int(t_h + c_h + 40)
                current_y += header_height
            
            current_page_num = 1
            
            # 페이지 컨테이너 생성 함수
            def create_page_widget(page_num):
                widget = QWidget()
                widget.setFixedWidth(PREVIEW_WIDTH)
                widget.setStyleSheet("background-color: white; border: 1px solid #999; margin-bottom: 20px;")
                
                main_layout = QVBoxLayout(widget)
                main_layout.setContentsMargins(0, 0, 0, 0)
                main_layout.setSpacing(0)
                
                # 상단 식별자
                lbl = QLabel(f"Page {page_num}")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("background-color: #eee; color: #555; font-size: 10px; padding: 2px;")
                main_layout.addWidget(lbl)
                
                # 콘텐츠 영역
                content_widget = QWidget()
                content_layout = QVBoxLayout(content_widget)
                content_layout.setContentsMargins(display_margin, display_margin, display_margin, display_margin)
                content_layout.setSpacing(display_spacing)
                content_layout.setAlignment(Qt.AlignTop)
                main_layout.addWidget(content_widget)
                
                # 페이지 번호 미리보기
                if page_num_pos_str != "없음":
                    num_lbl = QLabel(str(page_num))
                    num_lbl.setStyleSheet("font-weight: bold; color: black;")
                    
                    if "상단" in page_num_pos_str:
                        h_layout = QHBoxLayout()
                        h_layout.setContentsMargins(display_margin, 5, display_margin, 0)
                        if "우측" in page_num_pos_str: h_layout.addStretch(); h_layout.addWidget(num_lbl)
                        main_layout.insertLayout(1, h_layout)
                    elif "하단" in page_num_pos_str:
                        h_layout = QHBoxLayout()
                        h_layout.setContentsMargins(display_margin, 0, display_margin, 5)
                        if "우측" in page_num_pos_str: h_layout.addStretch(); h_layout.addWidget(num_lbl)
                        elif "중앙" in page_num_pos_str: h_layout.addStretch(); h_layout.addWidget(num_lbl); h_layout.addStretch()
                        main_layout.addLayout(h_layout)

                return widget, content_layout

            page_widget, content_layout = create_page_widget(current_page_num)
            self.preview_layout.addWidget(page_widget)

            # 제목/작곡가 미리보기 추가
            if title or composer:
                header_widget = QWidget()
                header_layout = QVBoxLayout(header_widget)
                header_layout.setContentsMargins(0, 0, 0, int(20*scale))
                header_layout.setSpacing(int(5*scale))

                if title:
                    title_label = QLabel(title)
                    title_label.setAlignment(Qt.AlignCenter)
                    title_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: black; border: none;")
                    header_layout.addWidget(title_label)
                if composer:
                    composer_label = QLabel(composer)
                    composer_label.setAlignment(Qt.AlignRight)
                    composer_label.setStyleSheet(f"font-size: 12px; color: #333; border: none;")
                    header_layout.addWidget(composer_label)
                
                content_layout.addWidget(header_widget)

            for path in file_paths:
                if not os.path.exists(path):
                    continue
                
                img = Image.open(path)
                
                # PDF 기준 콘텐츠 너비
                pdf_content_w = base_width - (margin * 2)
                if pdf_content_w < 1: pdf_content_w = 1
                
                # PDF 기준 높이 계산
                if img.width != pdf_content_w:
                    new_h = int(img.height * (pdf_content_w / img.width))
                else:
                    new_h = img.height
                
                # 페이지 넘김 체크
                if current_y + new_h + margin > page_height:
                    current_page_num += 1
                    current_y = margin
                    page_widget, content_layout = create_page_widget(current_page_num)
                    self.preview_layout.addWidget(page_widget)
                
                # 이미지 추가
                pix = QPixmap(path)
                if not pix.isNull():
                    lbl_img = QLabel()
                    scaled = pix.scaledToWidth(display_content_width, Qt.SmoothTransformation)
                    lbl_img.setPixmap(scaled)
                    lbl_img.setAlignment(Qt.AlignCenter)
                    content_layout.addWidget(lbl_img)
                
                current_y += new_h + spacing

        except Exception as e:
            print(f"Preview error: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Score Capture Pro - YouTube 악보 자동 캡처")
        self.resize(1300, 750)
        self.capture_area_dict = None
        self.captured_files = []
        self.capture_counter = 0
        self.is_capturing = False  # 캡처 상태 추적

        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.perform_capture)
        
        self.ad_block_timer = QTimer(self)
        self.ad_block_timer.timeout.connect(self.remove_youtube_ads)
        self.ad_block_timer.start(1000)

        self.last_captured_gray = None
        self.last_hash = None
        self.countdown_value = -1
        self.scroll_buffer = None
        self.current_scroll_filename = None

        self.setup_ui()
        self.apply_stylesheet()

    def apply_stylesheet(self):
        self.setStyleSheet(MODERN_STYLESHEET)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- 왼쪽 패널 ---
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setMaximumWidth(320)
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)
        
        # 헤더
        header_label = QLabel("Score Capture Pro")
        header_label.setObjectName("headerLabel")
        header_label.setAlignment(Qt.AlignLeft)
        left_layout.addWidget(header_label)
        
        # 1. URL 입력
        url_group = QGroupBox("YouTube URL")
        url_layout = QVBoxLayout()
        url_layout.setSpacing(6)
        url_layout.setContentsMargins(8, 8, 8, 8)
        
        url_h_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("유튜브 링크")
        self.url_input.setMinimumHeight(32)
        
        btn_go = QPushButton("이동")
        btn_go.setMinimumHeight(32)
        btn_go.setMaximumWidth(60)
        btn_go.clicked.connect(self.load_url)
        
        url_h_layout.addWidget(self.url_input)
        url_h_layout.addWidget(btn_go)
        url_layout.addLayout(url_h_layout)
        url_group.setLayout(url_layout)
        left_layout.addWidget(url_group)

        # 2. 설정 + 제어 통합
        control_group = QGroupBox("캡처 설정 및 제어")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(6)
        control_layout.setContentsMargins(8, 8, 8, 8)
        
        # 모드 선택
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("모드:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["페이지 넘김 (기본)", "가로 스크롤 (이어붙이기)"])
        mode_layout.addWidget(self.mode_combo, 1)
        control_layout.addLayout(mode_layout)

        # 설정 (가로 배치)
        settings_h = QHBoxLayout()
        settings_h.addWidget(QLabel("민감도:"))
        self.sensitivity_input = QLineEdit("0.9")
        self.sensitivity_input.setMaximumWidth(50)
        self.sensitivity_input.setMinimumHeight(28)
        settings_h.addWidget(self.sensitivity_input)
        
        settings_h.addWidget(QLabel("딜레이:"))
        self.delay_input = QLineEdit("3")
        self.delay_input.setMaximumWidth(50)
        self.delay_input.setMinimumHeight(28)
        settings_h.addWidget(self.delay_input)
        settings_h.addWidget(QLabel("초"))
        settings_h.addStretch()
        
        control_layout.addLayout(settings_h)
        
        # 버튼들
        self.btn_select = QPushButton("1. 영역 선택")
        self.btn_select.setObjectName("selectButton")
        self.btn_select.setMinimumHeight(36)
        self.btn_select.clicked.connect(self.toggle_selection_mode)
        
        # 캡처 토글 버튼 (시작/중지 통합)
        self.btn_capture = QPushButton("2. 캡처 시작")
        self.btn_capture.setObjectName("captureButton")
        self.btn_capture.setMinimumHeight(42)
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.toggle_capture)

        control_layout.addWidget(self.btn_select)
        control_layout.addWidget(self.btn_capture)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # 3. 캡처 목록 + 미리보기 통합
        capture_group = QGroupBox("캡처된 이미지")
        capture_layout = QVBoxLayout()
        capture_layout.setSpacing(6)
        capture_layout.setContentsMargins(8, 8, 8, 8)
        
        # 미리보기 (위)
        preview_section = QLabel("미리보기")
        preview_section.setObjectName("sectionLabel")
        capture_layout.addWidget(preview_section)
        
        self.image_preview_label = QLabel("선택된 이미지 없음")
        self.image_preview_label.setObjectName("previewLabel")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumHeight(140)
        self.image_preview_label.setMaximumHeight(140)
        capture_layout.addWidget(self.image_preview_label)
        
        # 목록 (아래)
        list_section = QLabel("목록")
        list_section.setObjectName("sectionLabel")
        capture_layout.addWidget(list_section)
        
        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(100)
        self.list_widget.setMaximumHeight(120)
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemClicked.connect(self.show_image_preview)
        self.list_widget.model().rowsMoved.connect(self.on_list_order_changed)
        self.list_widget.model().rowsRemoved.connect(self.on_list_order_changed)
        capture_layout.addWidget(self.list_widget)
        
        self.btn_delete = QPushButton("선택 삭제")
        self.btn_delete.setObjectName("deleteButton")
        self.btn_delete.setMinimumHeight(28)
        self.btn_delete.clicked.connect(self.delete_selected_item)
        capture_layout.addWidget(self.btn_delete)
        
        capture_group.setLayout(capture_layout)
        left_layout.addWidget(capture_group)

        # 4. PDF 생성
        self.btn_pdf = QPushButton("3. 편집 및 저장")
        self.btn_pdf.setObjectName("pdfButton")
        self.btn_pdf.setMinimumHeight(42)
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self.switch_to_editor)
        left_layout.addWidget(self.btn_pdf)

        # 5. 상태
        self.status_label = QLabel("준비 완료")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(32)
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        # --- 오른쪽 패널 (웹뷰) ---
        self.right_stack = QStackedWidget()
        
        # 페이지 0: 웹뷰
        webview_container = QWidget()
        webview_layout = QVBoxLayout(webview_container)
        webview_layout.setContentsMargins(0, 0, 0, 0)
        
        self.webview = QWebEngineView()
        self.webview.setUrl(QUrl("https://www.youtube.com"))
        webview_layout.addWidget(self.webview)
        
        # 페이지 1: 에디터
        self.editor_widget = ScoreEditorWidget()
        self.editor_widget.save_requested.connect(self.generate_pdf_final)
        self.editor_widget.cancel_requested.connect(self.switch_to_capture)
        
        self.right_stack.addWidget(webview_container)
        self.right_stack.addWidget(self.editor_widget)

        self.overlay = SelectionOverlay(self.webview)
        self.overlay.selection_finished.connect(self.finish_selection)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.right_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1000])
        
        main_layout.addWidget(splitter)

    def remove_youtube_ads(self):
        """유튜브 광고 제거"""
        ad_script = """
        (function() {
            const selectors = [
                '.video-ads', '.ytp-ad-module', '.ytp-ad-overlay-container',
                '#masthead-ad', '#player-ads', 'ytd-promoted-sparkles-renderer',
                'ytd-display-ad-renderer', '.ad-container', '.ad-div'
            ];
            selectors.forEach(s => {
                document.querySelectorAll(s).forEach(el => el.remove());
            });

            const skipButtons = ['.ytp-ad-skip-button', '.ytp-ad-skip-button-modern', '.ytp-skip-ad-button'];
            skipButtons.forEach(s => {
                const btn = document.querySelector(s);
                if(btn) btn.click();
            });

            const video = document.querySelector('video');
            if (video && document.querySelector('.ad-showing')) {
                if (!isNaN(video.duration)) {
                    video.currentTime = video.duration - 0.1;
                }
            }
        })();
        """
        self.webview.page().runJavaScript(ad_script)

    def set_youtube_state(self, action, value=None):
        if action == "play":
            self.webview.page().runJavaScript("document.querySelector('video').play();")
        elif action == "pause":
            self.webview.page().runJavaScript("document.querySelector('video').pause();")
        elif action == "speed":
            self.webview.page().runJavaScript(f"document.querySelector('video').playbackRate = {value};")
        elif action == "quality":
            js = "var p = document.getElementById('movie_player'); if(p){var l=p.getAvailableQualityLevels(); p.setPlaybackQualityRange(l[0]);}"
            self.webview.page().runJavaScript(js)

    def load_url(self):
        url = self.url_input.text().strip()
        if url:
            self.webview.setUrl(QUrl(url if url.startswith("http") else "https://"+url))
            self.status_label.setText("동영상 로드 완료")

    def toggle_selection_mode(self):
        if self.overlay.isVisible() and self.overlay.mode_active:
            self.overlay.set_active(False)
            self.overlay.hide()
            self.btn_select.setText("1. 영역 선택")
            self.status_label.setText("선택 취소됨")
        else:
            self.overlay.resize(self.webview.size())
            self.overlay.set_active(True)
            self.btn_select.setText("선택 취소")
            self.status_label.setText("영역을 드래그하세요")
            self.overlay.setFocus()

    def finish_selection(self, area_dict):
        self.capture_area_dict = area_dict
        self.btn_capture.setEnabled(True)
        self.btn_select.setText("1. 영역 선택")
        self.status_label.setText(f"영역 설정됨 ({area_dict['width']}×{area_dict['height']})")
        
        msg = QMessageBox(self)
        msg.setWindowTitle("영역 선택 완료")
        msg.setText(f"영역이 설정되었습니다.\n\n크기: {area_dict['width']} × {area_dict['height']} px")
        msg.setIcon(QMessageBox.Information)
        msg.setStyleSheet(MODERN_STYLESHEET)
        msg.exec_()

    def switch_to_editor(self):
        """에디터 모드로 전환"""
        files = self.get_ordered_files()
        if not files:
            return
        
        self.set_youtube_state("pause")
        self.editor_widget.load_preview(files)
        self.right_stack.setCurrentIndex(1)
        self.status_label.setText("PDF 편집 모드")
        self.btn_select.setEnabled(False)
        self.btn_capture.setEnabled(False)

    def switch_to_capture(self):
        """캡처 모드로 복귀"""
        self.right_stack.setCurrentIndex(0)
        self.status_label.setText("캡처 모드")
        self.btn_select.setEnabled(True)
        self.btn_capture.setEnabled(True)

    def toggle_capture(self):
        """캡처 시작/중지 토글"""
        if not self.is_capturing:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        self.is_capturing = True
        self.btn_capture.setText("■ 캡처 중지")
        self.btn_capture.setObjectName("captureButtonActive")
        self.btn_capture.setStyleSheet("")  # 스타일 재적용
        self.apply_stylesheet()
        
        self.btn_select.setEnabled(False)
        self.btn_pdf.setEnabled(False)
        
        self.overlay.set_lock(True)
        self.overlay.show()

        self.captured_files = []
        self.list_widget.clear()
        self.image_preview_label.setText("캡처 진행 중...")
        self.last_captured_gray = None
        self.last_hash = None
        self.scroll_buffer = None
        self.current_scroll_filename = None
        
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        self.countdown_value = int(self.delay_input.text())
        self.run_countdown()

    def run_countdown(self):
        if self.countdown_value > 0:
            self.status_label.setText(f"{self.countdown_value}초 후 시작...")
            self.countdown_value -= 1
            QTimer.singleShot(1000, self.run_countdown)
        else:
            self.status_label.setText("캡처 진행 중")
            self.set_youtube_state("quality")
            self.set_youtube_state("speed", 2.0)
            self.set_youtube_state("play")
            self.capture_timer.start(1000)

    def stop_capture(self):
        self.is_capturing = False
        self.capture_timer.stop()
        self.set_youtube_state("pause")
        self.overlay.set_lock(False)
        
        self.btn_capture.setText("2. 캡처 시작")
        self.btn_capture.setObjectName("captureButton")
        self.btn_capture.setStyleSheet("")  # 스타일 재적용
        self.apply_stylesheet()
        
        # 스크롤 모드: 캡처 종료 시 일괄 자르기 수행
        if self.mode_combo.currentIndex() == 1 and self.scroll_buffer is not None:
            self.status_label.setText("이미지 분석 및 자르는 중...")
            QApplication.processEvents()
            
            full_img = self.scroll_buffer
            total_width = full_img.shape[1]
            target_w = self.capture_area_dict['width']
            
            current_x = 0
            
            while current_x < total_width:
                # 남은 길이가 목표 너비의 1.2배 이하면 마지막 조각으로 저장
                if total_width - current_x <= target_w * 1.2:
                    page_img = full_img[:, current_x:]
                    if page_img.shape[1] > 50: # 너무 작은 조각 제외
                        self._save_image_to_list(page_img)
                    break
                
                # 자를 위치 탐색 (목표 너비의 80% ~ 120% 사이)
                search_min = int(target_w * 0.8)
                search_max = int(target_w * 1.2)
                
                # 전체 이미지 범위 내에서 탐색
                cut_x = self.find_best_cut_point(full_img, current_x + search_min, current_x + search_max)
                
                if cut_x:
                    page_img = full_img[:, current_x:cut_x]
                    self._save_image_to_list(page_img)
                    current_x = cut_x
                else:
                    # 적절한 자를 위치를 못 찾으면 강제로 target_w 만큼 자름
                    cut_x = current_x + target_w
                    page_img = full_img[:, current_x:cut_x]
                    self._save_image_to_list(page_img)
                    current_x = cut_x

            self.scroll_buffer = None

        self.btn_select.setEnabled(True)
        self.btn_pdf.setEnabled(len(self.captured_files) > 0)
        
        count = len(self.captured_files)
        self.status_label.setText(f"캡처 중지 (총 {count}개)")

    def _save_image_to_list(self, img):
        """이미지를 저장하고 리스트에 추가하는 내부 함수"""
        self.capture_counter += 1
        filename = os.path.join(OUTPUT_FOLDER, f"score_scroll_{self.capture_counter:03d}.png")
        cv2.imwrite(filename, img)
        self.captured_files.append(filename)
        
        item = QListWidgetItem(os.path.basename(filename))
        item.setData(Qt.UserRole, filename)
        self.list_widget.addItem(item)
        self.list_widget.scrollToBottom()
        self.display_image(filename)

    def find_best_cut_point(self, img, min_x, max_x):
        """이미지에서 최적의 자르기 위치(세로선)를 찾습니다."""
        h, w = img.shape[:2]
        if min_x >= max_x or max_x > w:
            return None
            
        roi = img[:, min_x:max_x]
        
        # 1. 전처리: 그레이스케일 -> 이진화
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 악보는 흰 배경에 검은 선이므로, 반전시켜서 선을 찾음 (THRESH_BINARY_INV)
        # 임계값 200은 일반적인 악보에서 적절함
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 2. 세로선 추출 (Morphological Open)
        # 높이의 30% 이상 되는 세로선만 남김 (음표 기둥 등 제거 목적)
        line_height = max(int(h * 0.3), 10) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_height))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 3. 수직 투영 (Vertical Projection)
        col_sums = np.sum(vertical_lines, axis=0) / 255  # 픽셀 수로 변환
        
        # 4. 피크 검출 (마디 줄 후보)
        candidates = []
        # 최소 10픽셀 이상 높이의 선이 감지되어야 함
        min_line_pixels = line_height * 0.5 
        
        for x in range(1, len(col_sums) - 1):
            val = col_sums[x]
            if val > min_line_pixels:
                # 로컬 피크 확인 (주변보다 값이 크거나 같음)
                if val >= col_sums[x-1] and val >= col_sums[x+1]:
                    # 중복 피크 방지 (인접한 픽셀이 계속 피크일 경우 하나만)
                    if not candidates or (min_x + x) - candidates[-1][0] > 5:
                        candidates.append((min_x + x, val))
        
        if candidates:
            # 가장 확실한 선(가장 긴 선)을 선택하거나, 
            # 여러 개라면 검색 범위의 끝부분(오른쪽)에 가까운 것을 선호할 수 있음.
            # 여기서는 '가장 긴 선'을 최우선으로 하되, 비슷하면 오른쪽 선호
            
            # 정렬: 1순위 길이(내림차순), 2순위 x좌표(내림차순)
            candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
            
            best_x = candidates[0][0]
            return best_x + 10 # 마디 줄 포함하고 약간의 여백
            
        # 5. 마디 줄이 감지되지 않은 경우: 공백(흰색 영역) 찾기
        # 원본 gray 이미지에서 가장 어두운 성분이 적은 곳
        inverted_gray = 255 - gray
        col_sums_gray = np.sum(inverted_gray, axis=0)
        
        # 검색 범위 내에서 가장 흰 부분 찾기 (스무딩 적용)
        kernel_size = 5
        smoothed = np.convolve(col_sums_gray, np.ones(kernel_size)/kernel_size, mode='valid')
        
        if len(smoothed) > 0:
            min_val_idx = np.argmin(smoothed)
            return min_x + min_val_idx + (kernel_size // 2)
            
        return min_x + np.argmin(col_sums_gray)

    def perform_capture(self):
        if not self.capture_area_dict:
            return
            
        try:
            w, h = self.capture_area_dict['width'], self.capture_area_dict['height']
            source_top_left_global = QPoint(self.capture_area_dict['left'], self.capture_area_dict['top'])
            source_top_left_local = self.webview.mapFromGlobal(source_top_left_global)
            source_rect = QRect(source_top_left_local, QSize(w, h))

            img = QImage(QSize(w, h), QImage.Format_ARGB32_Premultiplied)
            self.overlay.hide()
            painter = QPainter(img)
            self.webview.render(painter, QPoint(), QRegion(source_rect))
            painter.end()
            self.overlay.show()

            ptr = img.constBits()
            ptr.setsize(img.sizeInBytes())
            arr = np.array(ptr).reshape(h, w, 4)
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            
            # 모드에 따른 분기
            if self.mode_combo.currentIndex() == 0:
                # --- 페이지 넘김 모드 (기존 로직) ---
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                should_save = False
                threshold = float(self.sensitivity_input.text())
                
                if self.last_captured_gray is None:
                    should_save = True
                else:
                    score, _ = compare_ssim(self.last_captured_gray, img_gray, full=True)
                    if score < threshold:
                        should_save = True

                if should_save:
                    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    curr_hash = imagehash.phash(pil_img)
                    
                    if self.last_hash is None or (curr_hash - self.last_hash > 5):
                        self.capture_counter += 1
                        filename = os.path.join(OUTPUT_FOLDER, f"score_{self.capture_counter:03d}.png")
                        cv2.imwrite(filename, img_bgr)
                        self.captured_files.append(filename)
                        
                        item = QListWidgetItem(os.path.basename(filename))
                        item.setData(Qt.UserRole, filename)
                        self.list_widget.addItem(item)
                        self.list_widget.scrollToBottom()
                        
                        self.display_image(filename)
                        self.btn_pdf.setEnabled(True)

                        self.last_captured_gray = img_gray
                        self.last_hash = curr_hash
                        
                        count = len(self.captured_files)
                        self.status_label.setText(f"캡처 완료 (총 {count}개)")
            else:
                # --- 가로 스크롤 모드 (이어붙이기) ---
                if self.scroll_buffer is None:
                    self.scroll_buffer = img_bgr
                    # 첫 프레임은 버퍼링만 함
                    self.status_label.setText("스크롤 캡처 시작 (버퍼링...)")
                else:
                    # 템플릿 매칭으로 겹치는 부분 찾기
                    template_width = 200
                    if self.scroll_buffer.shape[1] >= template_width and img_bgr.shape[1] >= template_width:
                        template = self.scroll_buffer[:, -template_width:]
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        
                        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > 0.7: # 임계값
                            match_x = max_loc[0]
                            new_part_start = match_x + template_width
                            
                            if new_part_start < img_bgr.shape[1]:
                                new_part = img_bgr[:, new_part_start:]
                                if new_part.shape[1] > 0:
                                    self.scroll_buffer = np.hstack((self.scroll_buffer, new_part))
                                    self.status_label.setText(f"이어붙이기 중... (전체 폭: {self.scroll_buffer.shape[1]}px)")
                    
        except Exception as e:
            print(f"Capture Error: {e}")
            self.status_label.setText(f"캡처 오류")

    def show_image_preview(self, item):
        # UserRole에서 경로 가져오기
        path = item.data(Qt.UserRole)
        if not path:
            path = os.path.join(OUTPUT_FOLDER, item.text())
        self.display_image(path)

    def display_image(self, filepath):
        if os.path.exists(filepath):
            pixmap = QPixmap(filepath)
            scaled_pixmap = pixmap.scaled(
                self.image_preview_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_preview_label.setPixmap(scaled_pixmap)

    def get_ordered_files(self):
        """리스트 위젯의 순서대로 파일 경로 반환"""
        files = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path = item.data(Qt.UserRole)
            if path and os.path.exists(path):
                files.append(path)
        return files

    def delete_selected_item(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
            
        for item in items:
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)
            full_path = item.data(Qt.UserRole)
            
            if full_path in self.captured_files:
                self.captured_files.remove(full_path)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except: pass
            
        if self.list_widget.count() == 0:
            self.image_preview_label.clear()
            self.image_preview_label.setText("선택된 이미지 없음")
            self.btn_pdf.setEnabled(False)
        else:
            last_row = self.list_widget.count() - 1
            self.list_widget.setCurrentRow(last_row)
            self.show_image_preview(self.list_widget.item(last_row))
        
        self.status_label.setText(f"삭제 완료 (남은 이미지: {self.list_widget.count()}개)")

    def on_list_order_changed(self):
        if self.right_stack.currentIndex() == 1:
            files = self.get_ordered_files()
            self.editor_widget.load_preview(files)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self.list_widget.hasFocus():
            self.delete_selected_item()
        super().keyPressEvent(event)

    def create_pdf(self):
        self.switch_to_editor()

    def generate_pdf_final(self, metadata):
        files = self.get_ordered_files()
        if not files:
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "PDF 저장", 
            "악보_결과.pdf", 
            "PDF Files (*.pdf)"
        )
        
        if not path:
            return
        
        try:
            self.status_label.setText("PDF 생성 중...")
            QApplication.processEvents()
            
            # 설정값 파싱
            try:
                margin = int(metadata.get('margin', 60))
                spacing = int(metadata.get('spacing', 40))
            except ValueError:
                margin = 60
                spacing = 40
            
            page_num_pos = metadata.get('page_num_pos', '하단 중앙')

            image_objects = [Image.open(f).convert("RGB") for f in files]
            base_width = image_objects[0].width
            page_height = int(base_width * (297 / 210))
            
            # 여백을 고려한 콘텐츠 너비
            content_width = base_width - (margin * 2)
            if content_width < 100: content_width = base_width
            
            final_pages = []
            current_page = Image.new('RGB', (base_width, page_height), 'white')
            
            current_y = margin

            # 제목/작곡가 추가 (첫 페이지)
            title = metadata.get('title', '').strip()
            composer = metadata.get('composer', '').strip()
            
            if title or composer:
                draw = ImageDraw.Draw(current_page)
                try:
                    title_font = ImageFont.truetype("arial.ttf", size=int(base_width/30))
                    comp_font = ImageFont.truetype("arial.ttf", size=int(base_width/60))
                except:
                    title_font = ImageFont.load_default()
                    comp_font = ImageFont.load_default()
                
                header_offset = 0
                if title:
                    if hasattr(draw, "textbbox"):
                        bbox = draw.textbbox((0, 0), title, font=title_font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    else:
                        tw, th = draw.textsize(title, font=title_font)
                    draw.text(((base_width - tw) / 2, current_y), title, fill="black", font=title_font)
                    header_offset += th + 20
                
                if composer:
                    if hasattr(draw, "textbbox"):
                        bbox = draw.textbbox((0, 0), composer, font=comp_font)
                        cw = bbox[2] - bbox[0]
                        ch = bbox[3] - bbox[1]
                    else:
                        cw, ch = draw.textsize(composer, font=comp_font)
                    draw.text((base_width - margin - cw, current_y + header_offset), composer, fill="black", font=comp_font)
                    header_offset += ch + 20
                
                current_y += header_offset + 20

            for img in image_objects:
                if img.width != content_width:
                    new_h = int(img.height * (content_width / img.width))
                    img = img.resize((content_width, new_h), Image.Resampling.LANCZOS)
                else:
                    new_h = img.height
                    
                if current_y + new_h + margin > page_height:
                    final_pages.append(current_page)
                    current_page = Image.new('RGB', (base_width, page_height), 'white')
                    current_y = margin
                    
                current_page.paste(img, (margin, current_y))
                current_y += new_h + spacing
                
            final_pages.append(current_page)

            if page_num_pos != "없음":
                try:
                    draw_font = ImageFont.truetype("arial.ttf", size=max(14, int(base_width/50)))
                except IOError:
                    draw_font = ImageFont.load_default()

                for i, page in enumerate(final_pages, 1):
                    draw = ImageDraw.Draw(page)
                    text = f"{i} / {len(final_pages)}"
                    
                    if hasattr(draw, "textbbox"):
                        bbox = draw.textbbox((0, 0), text, font=draw_font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    else:
                        text_w, text_h = draw.textsize(text, font=draw_font)
                    
                    x_pos, y_pos = 0, 0
                    if page_num_pos == "하단 중앙":
                        x_pos = (base_width - text_w) // 2
                        y_pos = page_height - margin // 2 - text_h
                    elif page_num_pos == "하단 우측":
                        x_pos = base_width - margin - text_w
                        y_pos = page_height - margin // 2 - text_h
                    elif page_num_pos == "상단 우측":
                        x_pos = base_width - margin - text_w
                        y_pos = margin // 2
                        
                    draw.text((x_pos, y_pos), text, fill="black", font=draw_font)

            final_pages[0].save(path, save_all=True, append_images=final_pages[1:])
            
            self.status_label.setText(f"PDF 생성 완료 ({len(final_pages)}페이지)")
            
            msg = QMessageBox(self)
            msg.setWindowTitle("PDF 생성 완료")
            msg.setText(f"PDF가 생성되었습니다.\n\n총 페이지: {len(final_pages)}페이지")
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(MODERN_STYLESHEET)
            msg.exec_()
            self.switch_to_capture()
            
        except Exception as e:
            self.status_label.setText(f"PDF 생성 실패")
            QMessageBox.critical(self, "오류", f"PDF 생성 실패:\n{e}")

    def resizeEvent(self, event):
        if hasattr(self, 'overlay'):
            self.overlay.resize(self.webview.size())
        super().resizeEvent(event)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())