import sys
import os
import re
import time
import qrcode
import numpy as np
import cv2
import imagehash
from PIL import Image, ImageDraw, ImageFont

# --- PyQt5 라이브러리 ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,   
                             QGroupBox, QListWidget, QMessageBox, QFileDialog,
                             QSplitter, QFrame, QStackedWidget, QScrollArea, 
                             QFormLayout, QAbstractItemView, QListWidgetItem, QSlider,
                             QComboBox, QDialog, QCheckBox, QSizePolicy, QBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QSize, QPoint, QTimer, QEvent, QThread, QObject
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QFont, QFontDatabase

from skimage.metrics import structural_similarity as compare_ssim

# --- 설정 ---
OUTPUT_FOLDER = "captured_scores"

# --- 폰트 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE_DIR, "fonts")
FONT_BOLD_PATH = os.path.join(FONT_DIR, "NotoSansKR-Bold.ttf")
FONT_REGULAR_PATH = os.path.join(FONT_DIR, "NotoSansKR-Regular.ttf")

# --- 프로페셔널 스타일시트 ---
MODERN_STYLESHEET = """
/* 메인 윈도우 및 다이얼로그 (다크모드 대응) */
QMainWindow, QDialog {
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

/* 콤보박스 */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 4px 10px;
    color: #333333;
}

QComboBox:hover {
    border: 1px solid #0078d4;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #333333;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
    border: 1px solid #d0d0d0;
    outline: none;
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
    selection_cancelled = pyqtSignal()
    
    def __init__(self, parent=None): 
        super().__init__(parent)
        # 전체 화면 오버레이 설정
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        
        self.start_pos = None
        self.current_pos = None
        self.is_selecting = False
        self.bg_pixmap = None

    def start(self):
        """화면 캡처 후 선택 모드 시작"""
        desktop = QApplication.desktop()
        rect = desktop.geometry()
        self.setGeometry(rect)
        
        # 현재 화면을 캡처하여 배경으로 사용 (Freeze 효과)
        screen = QApplication.primaryScreen()
        self.bg_pixmap = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
        
        self.show()
        self.activateWindow()
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            self.selection_cancelled.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.current_pos = event.pos()
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.current_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            rect = QRect(self.start_pos, self.current_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                final_area = {
                    'top': self.y() + rect.y(),
                    'left': self.x() + rect.x(),
                    'width': rect.width(),
                    'height': rect.height()
                }
                self.close()
                self.selection_finished.emit(final_area)
            else:
                self.start_pos = None
                self.current_pos = None
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        if self.bg_pixmap:
            painter.drawPixmap(0, 0, self.bg_pixmap)
            
            # 배경 어둡게 처리
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
            
            if self.start_pos and self.current_pos:
                rect = QRect(self.start_pos, self.current_pos).normalized()
                
                # 선택 영역은 원본 밝기로 그리기
                painter.drawPixmap(rect, self.bg_pixmap, rect)
                
                # 테두리
                pen = QPen(QColor(0, 120, 212), 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(rect)
                
                # 사이즈 텍스트
                painter.setPen(Qt.white)
                font = QFont("Segoe UI", 9, QFont.Bold)
                painter.setFont(font)
                text = f"{rect.width()} × {rect.height()} px"
                painter.drawText(rect.topLeft() - QPoint(0, 5), text)
        else:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

class CaptureAreaIndicator(QWidget):
    """선택된 영역을 화면에 계속 표시하는 투명 위젯"""
    def __init__(self, x, y, w, h, parent=None):
        super().__init__(parent)
        self.setGeometry(x, y, w, h)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 마우스 이벤트를 통과시킴
        self.border_color = QColor(0, 255, 0)  # 기본: 초록색
        self.show()

    def set_color(self, color):
        self.border_color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.border_color, 3)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

def enhance_score_image(img_bgr):
    """이미지 선명화 및 이진화 처리 (스캔 품질)"""
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 2배 업스케일링 (Cubic Interpolation) - 선명도 확보
    enhanced = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 3. 적응형 이진화 (Adaptive Thresholding)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def cv2_to_qpixmap(img_cv):
    """OpenCV 이미지를 QPixmap으로 변환"""
    if img_cv is None:
        return QPixmap()
    # 메모리 연속성 보장
    if not img_cv.flags['C_CONTIGUOUS']:
        img_cv = np.ascontiguousarray(img_cv)
    
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def qpixmap_to_cv(pixmap):
    """QPixmap을 OpenCV 이미지(BGR)로 변환"""
    if pixmap.isNull():
        return None
    img = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.array(ptr).reshape(h, img.bytesPerLine())
    # 패딩 제거 및 BGR 변환
    arr = arr[:, :w * 3].reshape(h, w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def get_pil_font(path, size, fallback="arial.ttf"):
    """PIL 폰트 로드 헬퍼"""
    try:
        return ImageFont.truetype(path, size=size)
    except IOError:
        try:
            return ImageFont.truetype(fallback, size=size)
        except:
            return ImageFont.load_default()

def get_text_size(draw, text, font):
    """Pillow 버전 호환 텍스트 크기 계산"""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return draw.textsize(text, font=font)

def calculate_best_cut_point(img, min_x, max_x):
    """이미지에서 최적의 자르기 위치(세로선)를 찾습니다."""
    h, w = img.shape[:2]
    if min_x >= max_x or max_x > w:
        return None
        
    roi = img[:, min_x:max_x]
    
    # 1. 전처리: 그레이스케일 -> 이진화
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 세로선 추출
    line_height = max(int(h * 0.3), 10) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_height))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 3. 수직 투영
    col_sums = np.sum(vertical_lines, axis=0) / 255
    
    # 4. 피크 검출
    candidates = []
    min_line_pixels = line_height * 0.5 
    
    for x in range(1, len(col_sums) - 1):
        val = col_sums[x]
        if val > min_line_pixels:
            if val >= col_sums[x-1] and val >= col_sums[x+1]:
                if not candidates or (min_x + x) - candidates[-1][0] > 5:
                    candidates.append((min_x + x, val))
    
    if candidates:
        candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
        best_x = candidates[0][0]
        return best_x + 5
        
    # 5. 공백 찾기
    inverted_gray = 255 - gray
    col_sums_gray = np.sum(inverted_gray, axis=0)
    
    kernel_size = 5
    smoothed = np.convolve(col_sums_gray, np.ones(kernel_size)/kernel_size, mode='valid')
    
    if len(smoothed) > 0:
        min_val_idx = np.argmin(smoothed)
        return min_x + min_val_idx + (kernel_size // 2)
        
    return min_x + np.argmin(col_sums_gray)

class SlicerCanvas(QWidget):
    point_added = pyqtSignal(int)
    point_removed = pyqtSignal(int)

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.scale_factor = 1.0
        self.cut_points = []
        self.setFixedSize(pixmap.size())
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.hover_x = -1

    def set_cut_points(self, points):
        self.cut_points = sorted(list(set(points)))
        self.update()

    def perform_zoom(self, delta_y):
        if delta_y > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        
        new_size = self.pixmap.size() * self.scale_factor
        self.setFixedSize(new_size)
        self.update()

    def mouseMoveEvent(self, event):
        self.hover_x = int(event.pos().x() / self.scale_factor)
        self.update()

    def mousePressEvent(self, event):
        sx = event.pos().x()
        x = int(sx / self.scale_factor)
        x = max(0, min(x, self.pixmap.width()))
        
        if event.button() == Qt.LeftButton:
            self.cut_points.append(x)
            self.cut_points.sort()
            self.point_added.emit(x)
            self.update()
        elif event.button() == Qt.RightButton:
            closest = -1
            min_dist_screen = 20
            for p in self.cut_points:
                p_screen = p * self.scale_factor
                dist = abs(p_screen - sx)
                if dist < min_dist_screen:
                    min_dist_screen = dist
                    closest = p
            if closest != -1:
                self.cut_points.remove(closest)
                self.point_removed.emit(closest)
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawPixmap(0, 0, self.pixmap)
        
        pen = QPen(QColor(255, 0, 0), 2.0 / self.scale_factor)
        painter.setPen(pen)
        h = self.pixmap.height()
        for x in self.cut_points:
            painter.drawLine(x, 0, x, h)
            
        if 0 <= self.hover_x < self.pixmap.width():
            pen_hover = QPen(QColor(0, 120, 212, 150), 1.0 / self.scale_factor, Qt.DashLine)
            painter.setPen(pen_hover)
            painter.drawLine(self.hover_x, 0, self.hover_x, h)

class ScrollSlicerDialog(QDialog):
    def __init__(self, image, target_width, parent=None):
        super().__init__(parent)
        self.setWindowTitle("스크롤 캡처 자르기 편집")
        self.resize(1200, 800)
        self.image = image
        self.target_width = target_width
        
        layout = QVBoxLayout(self)
        
        info_layout = QHBoxLayout()
        lbl_info = QLabel("✂️ 이미지를 클릭하여 자를 위치(빨간선)를 추가하세요. (우클릭: 삭제, Ctrl+휠: 확대/축소)")
        lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        info_layout.addWidget(lbl_info)
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setAlignment(Qt.AlignCenter)
        
        self.pixmap = cv2_to_qpixmap(image)
        self.canvas = SlicerCanvas(self.pixmap)
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll)
        
        self.scroll.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)

        btn_layout = QHBoxLayout()
        
        btn_auto = QPushButton("자동 감지 실행")
        btn_auto.setMinimumHeight(40)
        btn_auto.clicked.connect(self.run_auto_detect)
        
        btn_clear = QPushButton("모두 지우기")
        btn_clear.setMinimumHeight(40)
        btn_clear.clicked.connect(self.clear_all_points)
        
        self.lbl_count = QLabel("예상 조각 수: -개")
        self.lbl_count.setStyleSheet("font-size: 13px; font-weight: bold; color: #0078d4; margin-left: 15px;")
        
        btn_cancel = QPushButton("취소")
        btn_cancel.setMinimumHeight(40)
        btn_cancel.clicked.connect(self.reject)
        
        btn_save = QPushButton("자르기 및 저장")
        btn_save.setMinimumHeight(40)
        btn_save.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.accept)
        
        btn_layout.addWidget(btn_auto)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(self.lbl_count)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        
        layout.addLayout(btn_layout)
        
        self.canvas.point_added.connect(lambda _: self.update_slice_count())
        self.canvas.point_removed.connect(lambda _: self.update_slice_count())
        
        QTimer.singleShot(100, self.run_auto_detect)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                self.canvas.perform_zoom(event.angleDelta().y())
                return True
            else:
                self.scroll.horizontalScrollBar().setValue(
                    self.scroll.horizontalScrollBar().value() - event.angleDelta().y()
                )
                return True
        return super().eventFilter(source, event)

    def clear_all_points(self):
        self.canvas.set_cut_points([])
        self.update_slice_count()

    def run_auto_detect(self):
        points = []
        total_width = self.image.shape[1]
        current_x = 0
        
        while current_x < total_width:
            if total_width - current_x <= self.target_width * 1.2:
                break
            
            search_min = int(self.target_width * 0.8)
            search_max = int(self.target_width * 1.2)
            
            cut_x = calculate_best_cut_point(self.image, current_x + search_min, current_x + search_max)
            
            if cut_x:
                points.append(cut_x)
                current_x = cut_x
            else:
                current_x += self.target_width
                points.append(current_x)
        
        self.canvas.set_cut_points(points)
        self.update_slice_count()

    def update_slice_count(self):
        points = sorted(list(set(self.canvas.cut_points)))
        count = 0
        start_x = 0
        h, w = self.image.shape[:2]
        
        calc_points = points.copy()
        if not calc_points or calc_points[-1] < w:
            calc_points.append(w)
            
        for x in calc_points:
            if x > start_x:
                x = min(x, w)
                if x - start_x > 50:
                    count += 1
                start_x = x
        
        self.lbl_count.setText(f"예상 조각 수: {count}개")

    def get_sliced_images(self):
        points = sorted(list(set(self.canvas.cut_points)))
        images = []
        start_x = 0
        h, w = self.image.shape[:2]
        
        if not points or points[-1] < w:
            points.append(w)
            
        for x in points:
            if x > start_x:
                x = min(x, w)
                img_chunk = self.image[:, start_x:x]
                if img_chunk.shape[1] > 50:
                    images.append(img_chunk)
                start_x = x
        return images

class DraggableScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.OpenHandCursor)
        self.last_pos = QPoint()
        self.scale_factor = 1.0
        self.original_pixmap = None
        self.label = None

    def set_image(self, image_path, enhance=False):
        if enhance:
            img = cv2.imread(image_path)
            if img is not None:
                processed = enhance_score_image(img)
                self.original_pixmap = cv2_to_qpixmap(processed)
            else:
                self.original_pixmap = QPixmap()
        else:
            self.original_pixmap = QPixmap(image_path)
            
        if self.original_pixmap.isNull():
            return
        
        self.label = QLabel()
        self.label.setPixmap(self.original_pixmap)
        self.label.resize(self.original_pixmap.size())
        self.setWidget(self.label)
        self.scale_factor = 1.0

    def wheelEvent(self, event):
        if not self.original_pixmap:
            super().wheelEvent(event)
            return

        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        
        new_w = int(self.original_pixmap.width() * self.scale_factor)
        new_h = int(self.original_pixmap.height() * self.scale_factor)
        
        self.label.setPixmap(self.original_pixmap.scaled(
            new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.label.resize(new_w, new_h)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

class ImageDetailDialog(QDialog):
    def __init__(self, image_path, enhance=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("이미지 상세 보기 (드래그:이동, 휠:확대/축소)")
        self.resize(1000, 800)
        
        layout = QVBoxLayout(self)
        
        scroll = DraggableScrollArea()
        scroll.set_image(image_path, enhance)
        layout.addWidget(scroll)
        
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.close)
        btn_close.setMinimumHeight(40)
        layout.addWidget(btn_close)

class ScoreEditorWidget(QWidget):
    """PDF 생성 전 메타데이터 입력 및 미리보기 위젯"""
    save_requested = pyqtSignal(dict)
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_files = []
        self.font_bold = "Arial"
        self.font_regular = "Arial"
        
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.refresh_preview)
        
        # 메인 레이아웃 (세로 배치)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 10)
        main_layout.setSpacing(10)

        # --- 상단 설정 영역 (가로 배치) ---
        settings_container = QWidget()
        settings_layout = QHBoxLayout(settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(20)

        # 1. 메타데이터 입력
        info_group = QGroupBox("기본 정보")
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setSpacing(10)
        
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("노래 제목")
        self.title_edit.setMinimumHeight(30)
        
        self.composer_edit = QLineEdit()
        self.composer_edit.setPlaceholderText("아티스트")
        self.composer_edit.setMinimumHeight(30)

        self.bpm_edit = QLineEdit()
        self.bpm_edit.setPlaceholderText("BPM (예: 120)")
        self.bpm_edit.setMinimumHeight(30)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("URL (QR코드, 예: 유튜브 링크)")
        self.url_edit.setMinimumHeight(30)
        
        self.title_edit.textChanged.connect(self.trigger_refresh)
        self.composer_edit.textChanged.connect(self.trigger_refresh)
        self.bpm_edit.textChanged.connect(self.trigger_refresh)
        self.url_edit.textChanged.connect(self.trigger_refresh)
        
        form_layout.addRow("제목:", self.title_edit)
        form_layout.addRow("아티스트:", self.composer_edit)
        form_layout.addRow("BPM:", self.bpm_edit)
        form_layout.addRow("URL (QR):", self.url_edit)
        info_group.setLayout(form_layout)
        settings_layout.addWidget(info_group)
        
        # 2. PDF 설정
        settings_group = QGroupBox("레이아웃 설정")
        settings_form = QFormLayout()
        settings_form.setSpacing(10)
        
        self.margin_edit = QLineEdit("60")
        self.margin_edit.setMinimumHeight(30)
        self.spacing_edit = QLineEdit("40")
        self.spacing_edit.setMinimumHeight(30)
        
        self.page_num_pos = QComboBox()
        self.page_num_pos.addItems(["하단 중앙", "하단 우측", "상단 우측", "없음"])
        self.page_num_pos.setMinimumHeight(30)
        
        self.chk_enhance = QCheckBox("화질 개선 (선명하게)")
        self.chk_enhance.setChecked(False)
        self.chk_enhance.stateChanged.connect(self.refresh_preview)
        
        settings_form.addRow("여백 (px):", self.margin_edit)
        settings_form.addRow("간격 (px):", self.spacing_edit)
        settings_form.addRow("페이지 번호:", self.page_num_pos)
        settings_form.addRow("옵션:", self.chk_enhance)
        settings_group.setLayout(settings_form)
        settings_layout.addWidget(settings_group)
        
        self.margin_edit.textChanged.connect(self.trigger_refresh)
        self.spacing_edit.textChanged.connect(self.trigger_refresh)
        self.page_num_pos.currentIndexChanged.connect(self.refresh_preview)

        main_layout.addWidget(settings_container)

        # --- 중앙 미리보기 영역 ---
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_frame.setStyleSheet("background-color: #525659; border: 1px solid #444;")
        
        preview_layout_inner = QVBoxLayout(preview_frame)
        preview_layout_inner.setContentsMargins(0, 0, 0, 0)
        preview_layout_inner.setSpacing(0)

        # 툴바
        preview_toolbar = QFrame()
        preview_toolbar.setStyleSheet("background-color: #333333; border-bottom: 1px solid #222;")
        preview_toolbar.setFixedHeight(40)
        tb_layout = QHBoxLayout(preview_toolbar)
        tb_layout.setContentsMargins(15, 0, 15, 0)
        
        lbl_preview_title = QLabel("미리보기")
        lbl_preview_title.setStyleSheet("color: #f0f0f0; font-weight: bold; font-size: 13px;")
        tb_layout.addWidget(lbl_preview_title)
        tb_layout.addStretch()
        lbl_guide = QLabel("이미지를 클릭하면 크게 볼 수 있습니다")
        lbl_guide.setStyleSheet("color: #aaa; font-size: 11px;")
        tb_layout.addWidget(lbl_guide)
        
        preview_layout_inner.addWidget(preview_toolbar)

        # 스크롤 영역
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background-color: #525659; }")
        
        self.preview_content = QWidget()
        self.preview_content.setStyleSheet("background-color: #525659;")
        self.preview_layout = QVBoxLayout(self.preview_content)
        self.preview_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.preview_layout.setSpacing(30)
        self.preview_layout.setContentsMargins(40, 40, 40, 40)
        
        self.scroll.setWidget(self.preview_content)
        preview_layout_inner.addWidget(self.scroll)
        
        main_layout.addWidget(preview_frame)

        # --- 하단 버튼 ---
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(10)
        
        self.btn_cancel = QPushButton("뒤로 가기")
        self.btn_cancel.setMinimumHeight(38)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        
        self.btn_save = QPushButton("저장하기")
        self.btn_save.setObjectName("captureButton")
        self.btn_save.setMinimumHeight(38)
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.clicked.connect(lambda: self.save_requested.emit({
            'title': self.title_edit.text(),
            'composer': self.composer_edit.text(),
            'bpm': self.bpm_edit.text(),
            'url': self.url_edit.text(),
            'margin': self.margin_edit.text(),
            'spacing': self.spacing_edit.text(),
            'page_num_pos': self.page_num_pos.currentText(),
            'enhance': self.chk_enhance.isChecked()
        }))
        
        btn_layout.addWidget(self.btn_cancel, 1)
        btn_layout.addWidget(self.btn_save, 1)
        
        main_layout.addLayout(btn_layout)

    def set_font_families(self, bold_family, regular_family):
        self.font_bold = bold_family
        self.font_regular = regular_family

    def reset_fields(self):
        self.title_edit.clear()
        self.composer_edit.clear()
        self.bpm_edit.clear()
        self.url_edit.clear()
        self.margin_edit.setText("60")
        self.spacing_edit.setText("40")
        self.page_num_pos.setCurrentIndex(0)
        self.chk_enhance.setChecked(False)
        self.current_files = []
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_large_image(self, path):
        if os.path.exists(path):
            enhance = self.chk_enhance.isChecked()
            dlg = ImageDetailDialog(path, enhance, self)
            dlg.exec_()

    def trigger_refresh(self):
        self.debounce_timer.start(500)

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

            # 화질 개선 시 여백/간격도 2배로 조정하여 비율 유지
            enhance_ratio = 1
            if self.chk_enhance.isChecked():
                margin *= 2
                spacing *= 2
                enhance_ratio = 2

            # 첫 번째 이미지로 기준 너비 설정 (PDF 생성 로직과 동일하게)
            # 화질 개선 여부에 따라 기준 너비가 달라짐
            first_img_cv = cv2.imread(file_paths[0])
            if first_img_cv is None: return
            
            if self.chk_enhance.isChecked():
                # 업스케일링 된 크기 예측 (2배)
                base_width = first_img_cv.shape[1] * 2
            else:
                base_width = first_img_cv.shape[1]
                
            # A4 비율 (210x297mm) -> 높이 계산
            page_height = int(base_width * (297 / 210))
            
            # 화면 표시용 스케일 (미리보기 너비 600px 기준)
            PREVIEW_WIDTH = 600
            scale = PREVIEW_WIDTH / base_width
            
            preview_height = int(page_height * scale)
            
            # 폰트 크기 계산 (PDF 생성 로직과 비율 맞춤)
            title_font_size = max(10, int((base_width / 30) * scale))
            comp_font_size = max(8, int((base_width / 60) * scale))
            
            current_y = margin
            
            title = self.title_edit.text().strip()
            composer = self.composer_edit.text().strip()
            bpm = self.bpm_edit.text().strip()
            url = self.url_edit.text().strip()
            
            current_page_num = 1
            
            # 페이지 컨테이너 생성 함수
            def create_page_widget(page_num):
                widget = QWidget()
                widget.setFixedSize(PREVIEW_WIDTH, preview_height)
                # 검정색 테두리 스타일
                widget.setStyleSheet("""
                    background-color: white; 
                    border: 1px solid black;
                """)
                
                # 상단 식별자
                lbl = QLabel(f"Page {page_num}", widget)
                lbl.setStyleSheet("background-color: transparent; color: #999; font-size: 10px; padding: 2px; border: none;")
                lbl.move(5, 5)
                lbl.show()
                
                return widget

            current_page_widget = create_page_widget(current_page_num)
            self.preview_layout.addWidget(current_page_widget)

            # 헤더 (제목/작곡가) 처리 - 절대 좌표 사용
            header_offset = 0
            qr_height_val = 0

            # QR 코드 생성 및 배치
            if url and qrcode:
                try:
                    qr = qrcode.QRCode(box_size=10, border=2)
                    qr.add_data(url)
                    qr.make(fit=True)
                    qr_img = qr.make_image(fill_color="black", back_color="white")
                    
                    # QPixmap 변환
                    data = qr_img.convert("RGBA").tobytes("raw", "RGBA")
                    qim = QImage(data, qr_img.size[0], qr_img.size[1], QImage.Format_RGBA8888)
                    qr_pix = QPixmap.fromImage(qim)
                    
                    # 크기 조정 (페이지 폭의 12%)
                    qr_size = int(base_width * 0.12)
                    qr_pix = qr_pix.scaled(qr_size, qr_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    lbl_qr = QLabel(current_page_widget)
                    lbl_qr.setPixmap(qr_pix)
                    lbl_qr.setScaledContents(True)
                    
                    # 위치: 좌측 상단 (여백 위치)
                    display_qr_size = int(qr_size * scale)
                    lbl_qr.setFixedSize(display_qr_size, display_qr_size)
                    lbl_qr.move(int(margin * scale), int(margin * scale))
                    lbl_qr.show()
                    
                    qr_height_val = qr_size
                except Exception as e:
                    print(f"QR Error: {e}")

            if title or composer or bpm:
                if title:
                    lbl_title = QLabel(title, current_page_widget)
                    lbl_title.setAlignment(Qt.AlignCenter)
                    font = QFont(self.font_bold, int(title_font_size/1.3), QFont.Bold)
                    lbl_title.setFont(font)
                    lbl_title.adjustSize()
                    
                    t_w = lbl_title.width()
                    t_h = lbl_title.height()
                    
                    x_pos = int((PREVIEW_WIDTH - t_w) / 2)
                    y_pos = int(current_y * scale)
                    
                    lbl_title.move(x_pos, y_pos)
                    lbl_title.show()
                    
                    header_offset += (t_h / scale) + (20 * enhance_ratio)

                if composer:
                    lbl_comp = QLabel(composer, current_page_widget)
                    lbl_comp.setAlignment(Qt.AlignRight)
                    font = QFont(self.font_regular, int(comp_font_size/1.3))
                    lbl_comp.setFont(font)
                    lbl_comp.adjustSize()
                    
                    c_w = lbl_comp.width()
                    c_h = lbl_comp.height()
                    
                    x_pos = int((base_width - margin) * scale) - c_w
                    y_pos = int((current_y + header_offset) * scale)
                    
                    lbl_comp.move(x_pos, y_pos)
                    lbl_comp.show()
                    
                    header_offset += (c_h / scale) + (20 * enhance_ratio)

                if bpm:
                    lbl_bpm = QLabel(f"BPM: {bpm}", current_page_widget)
                    lbl_bpm.setAlignment(Qt.AlignLeft)
                    font = QFont(self.font_bold, int(comp_font_size/1.3), QFont.Bold)
                    lbl_bpm.setFont(font)
                    lbl_bpm.adjustSize()
                    
                    b_w = lbl_bpm.width()
                    b_h = lbl_bpm.height()
                    
                    x_pos = int(margin * scale)
                    # QR 코드가 있고 높이가 겹치면 아래로 내리기 (좌측 정렬 유지)
                    if qr_height_val > 0 and header_offset < qr_height_val:
                        header_offset = qr_height_val + (10 * enhance_ratio)

                    y_pos = int((current_y + header_offset) * scale)
                    lbl_bpm.move(x_pos, y_pos)
                    lbl_bpm.show()
                    header_offset += (b_h / scale) + (10 * enhance_ratio)
                
            # 헤더 높이 결정 (텍스트와 QR 중 더 큰 것 기준)
            final_header_height = max(header_offset, qr_height_val)
            
            if final_header_height > 0:
                current_y += final_header_height + (60 * enhance_ratio)

            # 이미지 배치
            content_width_pdf = base_width - (margin * 2)
            if content_width_pdf < 1: content_width_pdf = 1
            
            display_content_width = int(content_width_pdf * scale)
            display_margin_left = int(margin * scale)

            for path in file_paths:
                if not os.path.exists(path):
                    continue
                
                # OpenCV로 로드하여 처리 후 QPixmap 변환
                img_cv = cv2.imread(path)
                if img_cv is None: continue
                
                if self.chk_enhance.isChecked():
                    img_cv = enhance_score_image(img_cv)
                
                pix = cv2_to_qpixmap(img_cv)
                
                img_w = pix.width()
                img_h = pix.height()
                
                if img_w != content_width_pdf:
                    new_h_pdf = int(img_h * (content_width_pdf / img_w))
                else:
                    new_h_pdf = img_h
                
                # 페이지 넘김 체크
                if current_y + new_h_pdf + margin > page_height:
                    current_page_num += 1
                    current_page_widget = create_page_widget(current_page_num)
                    self.preview_layout.addWidget(current_page_widget)
                    current_y = margin
                
                # 이미지 그리기
                display_h = int(new_h_pdf * scale)
                scaled_pix = pix.scaled(display_content_width, display_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                lbl_img = ClickableLabel(current_page_widget)
                lbl_img.setCursor(Qt.PointingHandCursor)
                lbl_img.setToolTip("클릭하여 크게 보기")
                lbl_img.clicked.connect(lambda p=path: self.show_large_image(p))
                lbl_img.setPixmap(scaled_pix)
                lbl_img.setFixedSize(display_content_width, display_h)
                lbl_img.move(display_margin_left, int(current_y * scale))
                lbl_img.show()
                
                current_y += new_h_pdf + spacing

            # 페이지 번호 추가
            if page_num_pos_str != "없음":
                total_pages = self.preview_layout.count()
                for i in range(total_pages):
                    page_widget = self.preview_layout.itemAt(i).widget()
                    if not page_widget: continue
                    
                    txt = f"{i+1} / {total_pages}"
                    lbl_num = QLabel(txt, page_widget)
                    font = QFont(self.font_regular, max(8, int((base_width/50)*scale/1.3)))
                    lbl_num.setFont(font)
                    lbl_num.adjustSize()
                    
                    nw = lbl_num.width()
                    nh = lbl_num.height()
                    
                    px_margin = int(margin * scale)
                    px_page_h = preview_height
                    
                    x_pos, y_pos = 0, 0
                    
                    if "하단" in page_num_pos_str:
                        y_pos = px_page_h - int(px_margin/2) - nh
                    elif "상단" in page_num_pos_str:
                        y_pos = int(px_margin/2)
                    
                    if "중앙" in page_num_pos_str:
                        x_pos = int((PREVIEW_WIDTH - nw) / 2)
                    elif "우측" in page_num_pos_str:
                        x_pos = PREVIEW_WIDTH - px_margin - nw
                        
                    lbl_num.move(x_pos, y_pos)
                    lbl_num.show()

        except Exception as e:
            print(f"Preview error: {e}")

class CaptureWorker(QObject):
    """캡처 및 이미지 처리를 담당하는 워커 스레드"""
    finished_processing = pyqtSignal()
    request_clean_capture = pyqtSignal()
    image_saved = pyqtSignal(str, object)  # filename, img_bgr
    scroll_updated = pyqtSignal(object)    # img_bgr
    status_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.last_captured_gray = None
        self.last_hash = None
        self.scroll_buffer = None
        self.capture_counter = 0

    def reset_state(self):
        self.last_captured_gray = None
        self.last_hash = None
        self.scroll_buffer = None
        self.capture_counter = 0

    def process_frame(self, img_bgr, mode_index, sensitivity):
        try:
            if mode_index == 0:  # 페이지 넘김 모드
                # 테두리 크롭 (비교 정확도 향상)
                h, w = img_bgr.shape[:2]
                border_crop = 5
                if w > 2 * border_crop and h > 2 * border_crop:
                    img_proc = img_bgr[border_crop:-border_crop, border_crop:-border_crop]
                else:
                    img_proc = img_bgr

                img_gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
                should_save = False

                if self.last_captured_gray is None:
                    should_save = True
                else:
                    if self.last_captured_gray.shape == img_gray.shape:
                        score, _ = compare_ssim(self.last_captured_gray, img_gray, full=True)
                        if score < sensitivity:
                            should_save = True
                    else:
                        should_save = True

                if should_save:
                    self.request_clean_capture.emit()
                    # 저장 완료될 때까지 finished_processing을 보내지 않음 (Main에서 save_clean 호출 대기)
                else:
                    self.finished_processing.emit()

            else:  # 스크롤 모드
                if self.scroll_buffer is None:
                    self.scroll_buffer = img_bgr
                    self.status_updated.emit("스크롤 캡처 시작 (버퍼링...)")
                    self.scroll_updated.emit(self.scroll_buffer)
                else:
                    # 템플릿 매칭
                    template_width = 200
                    if self.scroll_buffer.shape[1] >= template_width and img_bgr.shape[1] >= template_width:
                        template = self.scroll_buffer[:, -template_width:]
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)

                        if max_val > 0.7:
                            match_x = max_loc[0]
                            new_part_start = match_x + template_width
                            if new_part_start < img_bgr.shape[1]:
                                new_part = img_bgr[:, new_part_start:]
                                if new_part.shape[1] > 0:
                                    self.scroll_buffer = np.hstack((self.scroll_buffer, new_part))
                                    self.status_updated.emit(f"이어붙이기 중... (전체 폭: {self.scroll_buffer.shape[1]}px)")
                                    self.scroll_updated.emit(self.scroll_buffer)
                
                self.finished_processing.emit()
        except Exception as e:
            print(f"Worker Error: {e}")
            self.finished_processing.emit()

    def save_clean_image(self, img_bgr):
        try:
            # 해시 비교 (중복 저장 방지)
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            curr_hash = imagehash.phash(pil_img)

            if self.last_hash is None or (curr_hash - self.last_hash > 5):
                self.capture_counter += 1
                filename = os.path.join(OUTPUT_FOLDER, f"score_{self.capture_counter:03d}.png")
                cv2.imwrite(filename, img_bgr)
                
                # 다음 비교를 위해 크롭된 그레이스케일 저장
                h, w = img_bgr.shape[:2]
                border_crop = 5
                if w > 2 * border_crop and h > 2 * border_crop:
                    clean_proc = img_bgr[border_crop:-border_crop, border_crop:-border_crop]
                else:
                    clean_proc = img_bgr
                
                self.last_captured_gray = cv2.cvtColor(clean_proc, cv2.COLOR_BGR2GRAY)
                self.last_hash = curr_hash
                
                self.image_saved.emit(filename, img_bgr)
            
        except Exception as e:
            print(f"Save Error: {e}")
        finally:
            self.finished_processing.emit()

class MainWindow(QMainWindow):
    # 워커 스레드 통신용 시그널
    sig_process_frame = pyqtSignal(object, int, float)
    sig_save_clean = pyqtSignal(object)
    sig_reset_worker = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Score Capture Pro - 화면 악보 자동 캡처")
        self.resize(1200, 700)
        self.capture_area_dict = None
        self.captured_files = []
        self.capture_counter = 0
        self.is_capturing = False  # 캡처 상태 추적
        self.area_indicator = None # 선택 영역 표시 위젯
        self.current_original_pixmap = None

        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.perform_capture)

        self.countdown_value = -1
        self.current_scroll_buffer = None # UI 표시용 최신 버퍼
        self.current_scroll_filename = None
        self.is_worker_busy = False

        self.font_bold_family = "Arial"
        self.font_regular_family = "Arial"
        self.load_fonts()

        self.setup_ui()
        self.apply_stylesheet()
        self.setup_worker()

    def setup_worker(self):
        """워커 스레드 초기화"""
        self.worker_thread = QThread()
        self.worker = CaptureWorker()
        self.worker.moveToThread(self.worker_thread)
        
        # 시그널 연결
        self.sig_process_frame.connect(self.worker.process_frame)
        self.sig_save_clean.connect(self.worker.save_clean_image)
        self.sig_reset_worker.connect(self.worker.reset_state)
        
        self.worker.finished_processing.connect(self.on_worker_finished)
        self.worker.request_clean_capture.connect(self.on_request_clean_capture)
        self.worker.image_saved.connect(self.on_image_saved)
        self.worker.scroll_updated.connect(self.on_scroll_updated)
        self.worker.status_updated.connect(self.status_label.setText)
        
        self.worker_thread.start()

    def load_fonts(self):
        """폰트 파일 로드"""
        if os.path.exists(FONT_BOLD_PATH):
            id = QFontDatabase.addApplicationFont(FONT_BOLD_PATH)
            if id != -1:
                families = QFontDatabase.applicationFontFamilies(id)
                if families: self.font_bold_family = families[0]
        
        if os.path.exists(FONT_REGULAR_PATH):
            id = QFontDatabase.addApplicationFont(FONT_REGULAR_PATH)
            if id != -1:
                families = QFontDatabase.applicationFontFamilies(id)
                if families: self.font_regular_family = families[0]

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
        header_widget = QWidget()
        header_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        header_label = QLabel("Score Capture Pro")
        header_label.setObjectName("headerLabel")
        
        self.btn_mini = QPushButton("미니모드")
        self.btn_mini.setCheckable(True)
        self.btn_mini.setFixedSize(80, 32)
        self.btn_mini.setStyleSheet("font-size: 12px; background-color: #666; color: white; border-radius: 3px; padding: 0px;")
        self.btn_mini.clicked.connect(self.toggle_mini_mode)

        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_mini)
        left_layout.addWidget(header_widget)
        
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

        # 투명도 조절
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("투명도:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        opacity_layout.addWidget(self.opacity_slider)
        
        self.chk_always_on_top = QCheckBox("항상 위")
        self.chk_always_on_top.setStyleSheet("margin-left: 5px;")
        self.chk_always_on_top.stateChanged.connect(self.toggle_always_on_top)
        opacity_layout.addWidget(self.chk_always_on_top)
        control_layout.addLayout(opacity_layout)
        
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

        self.buttons_layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.buttons_layout.addWidget(self.btn_select)
        self.buttons_layout.addWidget(self.btn_capture)
        control_layout.addLayout(self.buttons_layout)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # 3. 캡처 목록 + 미리보기 통합
        self.capture_group = QGroupBox("캡처된 이미지")
        capture_layout = QVBoxLayout()
        capture_layout.setSpacing(6)
        capture_layout.setContentsMargins(8, 8, 8, 8)
        
        # 미리보기 라벨은 오른쪽 패널로 이동
        
        # 목록 (아래)
        list_section = QLabel("목록")
        list_section.setObjectName("sectionLabel")
        capture_layout.addWidget(list_section)
        
        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(100)
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemClicked.connect(self.show_image_preview)
        self.list_widget.model().rowsMoved.connect(self.on_list_order_changed)
        self.list_widget.model().rowsRemoved.connect(self.on_list_order_changed)
        capture_layout.addWidget(self.list_widget)
        
        # 버튼 레이아웃 (선택 삭제 / 전체 초기화)
        list_btn_layout = QHBoxLayout()
        
        self.btn_delete = QPushButton("선택 삭제")
        self.btn_delete.setObjectName("deleteButton")
        self.btn_delete.setMinimumHeight(28)
        self.btn_delete.clicked.connect(self.delete_selected_item)
        
        self.btn_reset = QPushButton("전체 초기화")
        self.btn_reset.setMinimumHeight(28)
        self.btn_reset.setStyleSheet("""
            QPushButton { background-color: #d9534f; color: white; border: none; border-radius: 4px; }
            QPushButton:hover { background-color: #c9302c; }
            QPushButton:pressed { background-color: #ac2925; }
        """)
        self.btn_reset.clicked.connect(self.reset_all)
        
        list_btn_layout.addWidget(self.btn_reset, 1)
        list_btn_layout.addWidget(self.btn_delete, 1)
        capture_layout.addLayout(list_btn_layout)
        
        self.capture_group.setLayout(capture_layout)
        left_layout.addWidget(self.capture_group, 1)

        # 미니 모드용 미리보기 (초기엔 숨김)
        self.mini_preview_label = QLabel()
        self.mini_preview_label.setAlignment(Qt.AlignCenter)
        self.mini_preview_label.setMinimumHeight(150)
        self.mini_preview_label.setStyleSheet("background-color: #f0f0f0; color: #666; border-radius: 4px; border: 1px solid #d0d0d0;")
        self.mini_preview_label.hide()
        left_layout.addWidget(self.mini_preview_label)

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

        # --- 오른쪽 패널 (대형 미리보기 및 에디터) ---
        self.right_stack = QStackedWidget()
        
        # 페이지 0: 캡처 대기/결과 화면
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(20, 20, 20, 20)
        
        self.image_preview_label = QLabel("영역을 선택하고 캡처를 시작하세요.\n캡처된 이미지가 여기에 표시됩니다.")
        self.image_preview_label.setObjectName("previewLabel")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setStyleSheet("background-color: #e0e0e0; border: 2px dashed #aaa; border-radius: 10px; font-size: 14px; color: #666;")
        self.image_preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        preview_layout.addWidget(QLabel("현재 캡처 미리보기", styleSheet="font-weight:bold; font-size:14px;"))
        preview_layout.addWidget(self.image_preview_label, 1)
        
        self.right_stack.addWidget(preview_container)
        
        # 페이지 1: 에디터
        self.editor_widget = ScoreEditorWidget()
        self.editor_widget.save_requested.connect(self.generate_pdf_final)
        self.editor_widget.cancel_requested.connect(self.switch_to_capture)
        self.editor_widget.set_font_families(self.font_bold_family, self.font_regular_family)
        
        self.right_stack.addWidget(self.editor_widget)

        self.overlay = SelectionOverlay()
        self.overlay.selection_finished.connect(self.finish_selection)
        self.overlay.selection_cancelled.connect(self.on_selection_cancelled)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.right_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1000])
        
        main_layout.addWidget(splitter)

    def toggle_mini_mode(self, checked):
        left_panel = self.findChild(QFrame, "leftPanel")

        if checked:
            self.right_stack.hide()
            self.capture_group.hide()
            self.mini_preview_label.show()
            self.btn_mini.setText("일반모드")
            
            if left_panel:
                left_panel.setStyleSheet("QFrame#leftPanel { border: none; background-color: #f5f5f5; }")
                left_panel.setMaximumWidth(16777215)
                left_panel.setMinimumWidth(0)
                if left_panel.layout():
                    left_panel.layout().setContentsMargins(5, 5, 5, 5)

            self.setFixedSize(320, 460)
            self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint | 
                              Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
            
            self.buttons_layout.setDirection(QBoxLayout.LeftToRight)
            
            self.btn_select.setText("1. 영역")
            self.btn_select.setFixedHeight(32)
            self.btn_select.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.btn_select.setStyleSheet("font-size: 12px; font-weight: bold; border-radius: 4px; border: none; color: white; background-color: #6a5acd;")
            
            if not self.is_capturing:
                self.btn_capture.setText("2. 캡처")
            else:
                self.btn_capture.setText("중지")
            self.btn_capture.setFixedHeight(32)
            self.btn_capture.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            bg_color = "#dc3545" if self.is_capturing else "#28a745"
            self.btn_capture.setStyleSheet(f"font-size: 12px; font-weight: bold; border-radius: 4px; border: none; color: white; background-color: {bg_color};")
            
            self.btn_pdf.setText("3. 편집 및 저장")
            self.status_label.setFixedHeight(28)
            self.status_label.setWordWrap(False)
            self.status_label.setStyleSheet("QLabel#statusLabel { padding: 0px 4px; font-size: 11px; background-color: #ffffff; border: 1px solid #d0d0d0; border-radius: 4px; color: #333333; }")
            self.show()
            self.raise_()
            self.activateWindow()
        else:
            self.right_stack.show()
            self.capture_group.show()
            self.mini_preview_label.hide()
            self.btn_mini.setText("미니모드")
            
            if left_panel:
                left_panel.setStyleSheet("")
                left_panel.setMaximumWidth(320)
                left_panel.setMinimumWidth(300)
                if left_panel.layout():
                    left_panel.layout().setContentsMargins(10, 10, 10, 10)
            
            self.setMinimumSize(1000, 600)
            self.setMaximumSize(16777215, 16777215)
            self.resize(1200, 700)
            
            # 일반 모드로 돌아올 때 '항상 위' 체크 여부 확인
            flags = Qt.Window
            if hasattr(self, 'chk_always_on_top') and self.chk_always_on_top.isChecked():
                flags |= Qt.WindowStaysOnTopHint
            self.setWindowFlags(flags)
            
            self.buttons_layout.setDirection(QBoxLayout.TopToBottom)
            
            self.btn_select.setText("1. 영역 선택")
            self.btn_select.setMinimumHeight(36)
            self.btn_select.setMaximumHeight(16777215)
            self.btn_select.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            self.btn_select.setStyleSheet("")
            
            if not self.is_capturing:
                self.btn_capture.setText("2. 캡처 시작")
            else:
                self.btn_capture.setText("■ 캡처 중지")
            self.btn_capture.setMinimumHeight(42)
            self.btn_capture.setMaximumHeight(16777215)
            self.btn_capture.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            self.btn_capture.setStyleSheet("")
            
            self.btn_pdf.setText("3. 편집 및 저장")
            self.status_label.setMinimumHeight(32)
            self.status_label.setMaximumHeight(16777215)
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("")
            self.show()
            self.raise_()
            self.activateWindow()
            
        self.update_mini_preview()

    def toggle_always_on_top(self, state):
        if self.btn_mini.isChecked():
            return # 미니모드는 이미 항상 위에 표시됨
            
        if state == Qt.Checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()
        self.raise_()
        self.activateWindow()

    def change_opacity(self, value):
        self.setWindowOpacity(value / 100.0)

    def toggle_selection_mode(self):
        self.switch_to_capture()
        # 메인 윈도우를 숨겨서 화면 전체를 선택할 수 있게 함
        self.hide()
        if self.area_indicator:
            self.area_indicator.close()
            self.area_indicator = None
        self.overlay.start()
        self.status_label.setText("영역 선택 중...")

    def finish_selection(self, area_dict):
        self.show() # 메인 윈도우 복구
        self.capture_area_dict = area_dict
        
        # 선택 영역 표시 위젯 생성
        if self.area_indicator:
            self.area_indicator.close()
        self.area_indicator = CaptureAreaIndicator(
            area_dict['left'], area_dict['top'], 
            area_dict['width'], area_dict['height']
        )
        
        self.btn_capture.setEnabled(True)
        if self.btn_mini.isChecked():
            self.btn_select.setText("1. 영역")
        else:
            self.btn_select.setText("1. 영역 선택")
        self.status_label.setText(f"영역 설정됨 ({area_dict['width']}×{area_dict['height']})")
        
    def on_selection_cancelled(self):
        self.show()
        if self.capture_area_dict:
            if self.area_indicator:
                self.area_indicator.close()
            self.area_indicator = CaptureAreaIndicator(
                self.capture_area_dict['left'], 
                self.capture_area_dict['top'], 
                self.capture_area_dict['width'], 
                self.capture_area_dict['height']
            )
            self.status_label.setText(f"영역 설정됨 ({self.capture_area_dict['width']}×{self.capture_area_dict['height']})")
        else:
            self.status_label.setText("준비 완료")

    def switch_to_editor(self):
        """에디터 모드로 전환"""
        if self.btn_mini.isChecked():
            self.btn_mini.setChecked(False)
            self.toggle_mini_mode(False)

        files = self.get_ordered_files()
        if not files:
            return
        
        self.editor_widget.load_preview(files)
        self.right_stack.setCurrentIndex(1)
        self.status_label.setText("PDF 편집 모드")

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
        self.switch_to_capture()
        self.is_capturing = True
        self.btn_capture.setObjectName("captureButtonActive")
        self.btn_capture.setStyleSheet("")  # 스타일 재적용
        self.apply_stylesheet()
        
        if self.btn_mini.isChecked():
            self.btn_capture.setText("중지")
            self.btn_capture.setStyleSheet("font-size: 12px; font-weight: bold; border-radius: 4px; border: none; color: white; background-color: #dc3545;")
        else:
            self.btn_capture.setText("■ 캡처 중지")
        
        if self.area_indicator:
            self.area_indicator.set_color(QColor(255, 0, 0)) # 빨간색 (녹화 중)
        
        self.btn_select.setEnabled(False)
        self.btn_pdf.setEnabled(False)
        
        self.captured_files = []
        self.list_widget.clear()
        self.image_preview_label.setText("캡처 진행 중...")
        self.current_original_pixmap = None
        self.update_mini_preview()
        self.current_scroll_buffer = None
        self.current_scroll_filename = None
        self.sig_reset_worker.emit()
        
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        self.countdown_value = int(self.delay_input.text())
        self.run_countdown()

    def run_countdown(self):
        if not self.is_capturing:
            return

        if self.countdown_value > 0:
            self.status_label.setText(f"{self.countdown_value}초 후 시작...")
            self.countdown_value -= 1
            QTimer.singleShot(1000, self.run_countdown)
        else:
            self.status_label.setText("캡처 진행 중")
            self.capture_timer.start(1000)

    def stop_capture(self):
        self.is_capturing = False
        self.capture_timer.stop()
        
        self.btn_capture.setObjectName("captureButton")
        self.btn_capture.setStyleSheet("")  # 스타일 재적용
        self.apply_stylesheet()
        
        if self.btn_mini.isChecked():
            self.btn_capture.setText("2. 캡처")
            self.btn_capture.setStyleSheet("font-size: 12px; font-weight: bold; border-radius: 4px; border: none; color: white; background-color: #28a745;")
        else:
            self.btn_capture.setText("2. 캡처 시작")
        
        if self.area_indicator:
            self.area_indicator.set_color(QColor(0, 255, 0)) # 초록색 (대기 중)
        
        # 스크롤 모드: 캡처 종료 시 일괄 자르기 수행
        if self.mode_combo.currentIndex() == 1 and self.current_scroll_buffer is not None:
            self.status_label.setText("편집 창을 여는 중...")
            QApplication.processEvents()
            
            dlg = ScrollSlicerDialog(self.current_scroll_buffer, self.capture_area_dict['width'], self)
            if dlg.exec_() == QDialog.Accepted:
                sliced_images = dlg.get_sliced_images()
                for img in sliced_images:
                    self._save_image_to_list(img)
                self.status_label.setText(f"스크롤 캡처 완료 ({len(sliced_images)}장)")
            else:
                self.status_label.setText("스크롤 캡처 취소됨")
            
            self.current_scroll_buffer = None

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

    def perform_capture(self):
        if not self.capture_area_dict or self.is_worker_busy:
            return
        
        self.is_worker_busy = True
        # 캡처 영역 크기 가져오기
        w, h = self.capture_area_dict['width'], self.capture_area_dict['height']
            
        try:
            is_scroll_mode = (self.mode_combo.currentIndex() == 1)

            # 스크롤 모드일 경우: 캡처 전 인디케이터 숨김
            if is_scroll_mode and self.area_indicator:
                self.area_indicator.hide()
                QApplication.processEvents()
                time.sleep(0.2)

            # 화면 캡처 (Global Coordinates)
            screen = QApplication.primaryScreen()
            # grabWindow(0)은 데스크탑 전체를 의미, 좌표는 글로벌 좌표
            pixmap = screen.grabWindow(0, self.capture_area_dict['left'], self.capture_area_dict['top'], w, h)
            
            # 스크롤 모드일 경우: 캡처 후 인디케이터 복구
            if is_scroll_mode and self.area_indicator:
                self.area_indicator.show()

            # QPixmap -> OpenCV 변환 (Main Thread)
            img_bgr = qpixmap_to_cv(pixmap)
            
            # 워커 스레드로 처리 위임
            mode = self.mode_combo.currentIndex()
            sensitivity = float(self.sensitivity_input.text())
            self.sig_process_frame.emit(img_bgr, mode, sensitivity)
            
        except Exception as e:
            print(f"Capture Error: {e}")
            self.status_label.setText(f"캡처 오류")
            self.is_worker_busy = False

    def on_worker_finished(self):
        """워커 처리 완료 시 호출"""
        self.is_worker_busy = False

    def on_request_clean_capture(self):
        """페이지 모드: 변화 감지 시 깨끗한 이미지 캡처 요청"""
        if not self.capture_area_dict:
            self.is_worker_busy = False
            return

        # 인디케이터 숨기고 캡처
        if self.area_indicator:
            self.area_indicator.hide()
            QApplication.processEvents()
            time.sleep(0.1)

        screen = QApplication.primaryScreen()
        pixmap_clean = screen.grabWindow(0, self.capture_area_dict['left'], self.capture_area_dict['top'], 
                                       self.capture_area_dict['width'], self.capture_area_dict['height'])
        
        if self.area_indicator:
            self.area_indicator.show()
        
        img_bgr_clean = qpixmap_to_cv(pixmap_clean)
        self.sig_save_clean.emit(img_bgr_clean)

    def on_image_saved(self, filename, img_bgr):
        """이미지 저장 완료 후 UI 업데이트"""
        self.captured_files.append(filename)
        item = QListWidgetItem(os.path.basename(filename))
        item.setData(Qt.UserRole, filename)
        self.list_widget.addItem(item)
        self.list_widget.scrollToBottom()
        
        self.display_image(filename)
        self.btn_pdf.setEnabled(True)
        
        count = len(self.captured_files)
        self.status_label.setText(f"캡처 완료 (총 {count}개)")

    def on_scroll_updated(self, scroll_buffer):
        """스크롤 모드: 버퍼 업데이트 시 미리보기 갱신"""
        self.current_scroll_buffer = scroll_buffer
        # 미리보기: 전체 이미지가 아닌 최근 캡처 영역만큼만 표시
        if self.capture_area_dict:
            w = self.capture_area_dict['width']
            display_w = min(scroll_buffer.shape[1], w)
            self.display_cv_image(scroll_buffer[:, -display_w:])

    def show_image_preview(self, item):
        # UserRole에서 경로 가져오기
        path = item.data(Qt.UserRole)
        if not path:
            path = os.path.join(OUTPUT_FOLDER, item.text())
        self.display_image(path)

    def display_cv_image(self, cv_img):
        """OpenCV 이미지를 미리보기 라벨에 표시"""
        self.current_original_pixmap = cv2_to_qpixmap(cv_img)
        self.update_preview_label()
        self.update_mini_preview()

    def display_image(self, filepath):
        if os.path.exists(filepath):
            self.current_original_pixmap = QPixmap(filepath)
            self.update_preview_label()
            self.update_mini_preview()

    def update_preview_label(self):
        if self.current_original_pixmap and not self.current_original_pixmap.isNull():
            # 라벨의 테두리 등을 고려하여 약간 작게 스케일링 (10px 여유)
            target_size = self.image_preview_label.size() - QSize(10, 10)
            if target_size.width() <= 0 or target_size.height() <= 0: target_size = QSize(1, 1)
            
            scaled_pixmap = self.current_original_pixmap.scaled(
                target_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_preview_label.setPixmap(scaled_pixmap)

    def update_mini_preview(self):
        if self.mini_preview_label.isVisible():
            if self.current_original_pixmap and not self.current_original_pixmap.isNull():
                target_size = self.mini_preview_label.size() - QSize(4, 4)
                scaled = self.current_original_pixmap.scaled(
                    target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.mini_preview_label.setPixmap(scaled)
            else:
                self.mini_preview_label.clear()
                self.mini_preview_label.setText("대기 중")

    def resizeEvent(self, event):
        self.update_preview_label()
        self.update_mini_preview()
        super().resizeEvent(event)

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
            
        reply = QMessageBox.question(self, '삭제 확인', f'선택한 {len(items)}개의 이미지를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
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
            self.current_original_pixmap = None
        else:
            last_row = self.list_widget.count() - 1
            self.list_widget.setCurrentRow(last_row)
            self.show_image_preview(self.list_widget.item(last_row))
        
        self.status_label.setText(f"삭제 완료 (남은 이미지: {self.list_widget.count()}개)")

    def reset_all(self):
        """모든 데이터 초기화 및 캡처 모드 복귀"""
        reply = QMessageBox.question(self, '초기화 확인', '모든 캡처 데이터를 삭제하고 초기화하시겠습니까?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.current_scroll_buffer = None
            self.stop_capture()
            self.captured_files = []
            self.list_widget.clear()
            self.image_preview_label.clear()
            self.image_preview_label.setText("영역을 선택하고 캡처를 시작하세요.\n캡처된 이미지가 여기에 표시됩니다.")
            self.current_original_pixmap = None
            self.sig_reset_worker.emit()
            self.update_mini_preview()
            self.btn_pdf.setEnabled(False)
            self.editor_widget.reset_fields()
            self.switch_to_capture()
            self.status_label.setText("초기화 완료")

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
            
        # 파일 이름 생성 로직
        title = metadata.get('title', '').strip()
        composer = metadata.get('composer', '').strip()
        bpm = metadata.get('bpm', '').strip()
        url = metadata.get('url', '').strip()
        
        if title or composer:
            if title and composer:
                filename_base = f"{title} - {composer} (TAB)"
            else:
                filename_base = title if title else composer
        else:
            # 유튜브 제목 사용 ( - YouTube 접미사 제거)
            page_title = "Captured Score"
            filename_base = page_title.replace(" - YouTube", "") if page_title else "TAB"
            
        # 파일명 특수문자 제거 (윈도우 파일명 금지 문자)
        filename_base = re.sub(r'[\\/*?:"<>|]', "", filename_base).strip()
        if not filename_base:
            filename_base = "TAB"
            
        path, filter_selected = QFileDialog.getSaveFileName(
            self, 
            "파일 저장", 
            filename_base, 
            "PDF Files (*.pdf);;PNG Images (*.png);;JPEG Images (*.jpg)"
        )
        
        if not path:
            return
            
        # 확장자 확인 및 보정
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            if "png" in filter_selected.lower():
                ext = ".png"
            elif "jpg" in filter_selected.lower():
                ext = ".jpg"
            else:
                ext = ".pdf"
            path += ext
        
        try:
            self.status_label.setText("파일 생성 중...")
            QApplication.processEvents()
            
            # 설정값 파싱
            try:
                margin = int(metadata.get('margin', 60))
                spacing = int(metadata.get('spacing', 40))
            except ValueError:
                margin = 60
                spacing = 40
            
            page_num_pos = metadata.get('page_num_pos', '하단 중앙')
            do_enhance = metadata.get('enhance', False)

            # 화질 개선 시 여백/간격도 2배로 조정하여 비율 유지
            enhance_ratio = 1
            if do_enhance:
                margin *= 2
                spacing *= 2
                enhance_ratio = 2

            image_objects = []
            for f in files:
                if do_enhance:
                    # OpenCV로 로드 -> 처리 -> PIL 변환
                    cv_img = cv2.imread(f)
                    if cv_img is not None:
                        processed = enhance_score_image(cv_img)
                        image_objects.append(Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)))
                else:
                    image_objects.append(Image.open(f).convert("RGB"))
            
            base_width = image_objects[0].width
            page_height = int(base_width * (297 / 210))
            
            # 여백을 고려한 콘텐츠 너비
            content_width = base_width - (margin * 2)
            if content_width < 100: content_width = base_width
            
            final_pages = []
            current_page = Image.new('RGB', (base_width, page_height), 'white')
            
            current_y = margin

            qr_height_val = 0
            if url and qrcode:
                try:
                    qr = qrcode.QRCode(box_size=10, border=2)
                    qr.add_data(url)
                    qr.make(fit=True)
                    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
                    
                    qr_size = int(base_width * 0.12)
                    qr_img = qr_img.resize((qr_size, qr_size), Image.Resampling.LANCZOS)
                    
                    current_page.paste(qr_img, (margin, margin))
                    qr_height_val = qr_size
                except Exception as e:
                    print(f"PDF QR Error: {e}")

            # 제목/작곡가 추가 (첫 페이지)
            title = metadata.get('title', '').strip()
            composer = metadata.get('composer', '').strip()
            bpm = metadata.get('bpm', '').strip()
            
            if title or composer or bpm:
                draw = ImageDraw.Draw(current_page)
                title_font = get_pil_font(FONT_BOLD_PATH, int(base_width/30))
                comp_font = get_pil_font(FONT_REGULAR_PATH, int(base_width/60))
                bpm_font = get_pil_font(FONT_BOLD_PATH, int(base_width/60))
                
                header_offset = 0
                if title:
                    tw, th = get_text_size(draw, title, title_font)
                    draw.text(((base_width - tw) / 2, current_y), title, fill="black", font=title_font)
                    header_offset += th + (20 * enhance_ratio)
                
                if composer:
                    cw, ch = get_text_size(draw, composer, comp_font)
                    draw.text((base_width - margin - cw, current_y + header_offset), composer, fill="black", font=comp_font)
                    header_offset += ch + (20 * enhance_ratio)
                
                if bpm:
                    bpm_text = f"BPM: {bpm}"
                    bw, bh = get_text_size(draw, bpm_text, bpm_font)
                    
                    if qr_height_val > 0 and header_offset < qr_height_val:
                        header_offset = qr_height_val + (10 * enhance_ratio)
                        
                    draw.text((margin, current_y + header_offset), bpm_text, fill="black", font=bpm_font)
                    header_offset += bh + (10 * enhance_ratio)

            final_header_height = max(header_offset, qr_height_val)
            if final_header_height > 0:
                current_y += final_header_height + (100 * enhance_ratio)

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
                draw_font = get_pil_font(FONT_REGULAR_PATH, max(14, int(base_width/50)))

                for i, page in enumerate(final_pages, 1):
                    draw = ImageDraw.Draw(page)
                    text = f"{i} / {len(final_pages)}"
                    text_w, text_h = get_text_size(draw, text, draw_font)
                    
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

            if ext == ".pdf":
                final_pages[0].save(path, save_all=True, append_images=final_pages[1:])
                msg_title = "PDF 생성 완료"
                msg_text = f"PDF가 생성되었습니다.\n\n총 페이지: {len(final_pages)}페이지"
            else:
                # 이미지로 저장 (여러 페이지일 경우 번호 붙임)
                base_path, _ = os.path.splitext(path)
                saved_count = 0
                for i, page in enumerate(final_pages):
                    if len(final_pages) > 1:
                        save_path = f"{base_path}_{i+1:02d}{ext}"
                    else:
                        save_path = path
                    
                    if ext in ['.jpg', '.jpeg']:
                        page.save(save_path, quality=95)
                    else:
                        page.save(save_path)
                    saved_count += 1
                
                msg_title = "이미지 저장 완료"
                msg_text = f"이미지가 저장되었습니다.\n\n저장된 파일 수: {saved_count}개"

            self.status_label.setText(f"저장 완료")
            
            msg = QMessageBox(self)
            msg.setWindowTitle(msg_title)
            msg.setText(msg_text)
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(MODERN_STYLESHEET)
            msg.exec_()
            self.switch_to_capture()
            
        except Exception as e:
            self.status_label.setText(f"저장 실패")
            QMessageBox.critical(self, "오류", f"파일 저장 실패:\n{e}")

    def closeEvent(self, event):
        if self.is_capturing:
            self.stop_capture() 
        if self.area_indicator:
            self.area_indicator.close()
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.close()
        if hasattr(self, 'worker_thread'):
            self.worker_thread.quit()
            self.worker_thread.wait()
        super().closeEvent(event)
        QApplication.instance().quit()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())