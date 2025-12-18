import sys
import os
import time
import numpy as np
import cv2
import mss
import imagehash
from PIL import Image, ImageDraw, ImageFont

# --- PyQt5 라이브러리 ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QGroupBox, QListWidget, QMessageBox, QFileDialog,
                             QSplitter, QFrame, QSizePolicy, QAbstractItemView)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QRect, QRectF, QSize, QPoint, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QPainterPath, QRegion

from skimage.metrics import structural_similarity as compare_ssim

# --- 설정 ---
OUTPUT_FOLDER = "captured_scores"



class SelectionOverlay(QWidget):
    """개선된 영역 선택 오버레이: 마스킹 효과 및 방향키 조절, 테두리 유지 기능"""
    selection_finished = pyqtSignal(dict) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.start_pos = None
        self.current_pos = None
        self.is_selecting = False
        self.confirmed_rect = None # 최종 선택된 영역 저장용
        self.mode_active = False   # 현재 드래그 가능한 선택 모드인지 여부
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.hide()

    def set_active(self, active):
        """선택 모드 활성화/비활성화 시 클릭 관통 설정"""
        self.mode_active = active
        if active:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False) # 클릭 받음
            self.setCursor(Qt.CrossCursor)
        else:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # 클릭 관통 (유튜브 조작용)
            self.setCursor(Qt.ArrowCursor)
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
                self.confirmed_rect = rect
                self.emit_selection()
            self.update()

    def keyPressEvent(self, event):
        """방향키로 영역 미세 조절 (1px 이동)"""
        if self.mode_active and self.confirmed_rect:
            if event.key() == Qt.Key_Left: self.confirmed_rect.translate(-1, 0)
            elif event.key() == Qt.Key_Right: self.confirmed_rect.translate(1, 0)
            elif event.key() == Qt.Key_Up: self.confirmed_rect.translate(0, -1)
            elif event.key() == Qt.Key_Down: self.confirmed_rect.translate(0, 1)
            self.update()
            self.emit_selection()

    def emit_selection(self):
        if self.confirmed_rect:
            global_top_left = self.mapToGlobal(self.confirmed_rect.topLeft())
            final_area = {
                'top': global_top_left.y(),
                'left': global_top_left.x(),
                'width': self.confirmed_rect.width(),
                'height': self.confirmed_rect.height()
            }
            self.selection_finished.emit(final_area)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        active_rect = None
        if self.is_selecting: active_rect = QRect(self.start_pos, self.current_pos).normalized()
        elif self.confirmed_rect: active_rect = self.confirmed_rect

        # 1. 배경 마스킹 (선택 모드일 때만 어둡게)
        if self.mode_active:
            overlay_color = QColor(0, 0, 0, 150)
            path = QPainterPath()
            path.addRect(QRectF(self.rect())) # 전체
            if active_rect:
                path.addRect(QRectF(active_rect)) # 선택된 곳 구멍 뚫기
            painter.fillPath(path, overlay_color)

        # 2. 테두리 그리기 (상시 표시)
        if active_rect:
            color = QColor(0, 174, 255) if self.mode_active else QColor(0, 255, 0) # 선택중 파랑, 고정 연두
            painter.setPen(QPen(color, 2, Qt.SolidLine))
            painter.drawRect(active_rect)
            
            if self.mode_active:
                painter.setPen(Qt.white)
                painter.drawText(active_rect.topLeft() + QPoint(5, -10), f"{active_rect.width()}x{active_rect.height()} (방향키 조절)")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube 악보 캡처 Pro (PyQt5)")
        self.resize(1300, 850)
        self.capture_area_dict = None
        self.captured_files = []

        # --- 캡처 로직 관련 멤버 변수 ---
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.perform_capture)
        self.last_captured_gray = None
        self.last_hash = None
        self.countdown_value = -1

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- 왼쪽 패널 (컨트롤) - 원래 디자인 유지 ---
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setMaximumWidth(400)
        left_panel.setMinimumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # 1. URL 입력
        url_group = QGroupBox("YouTube URL")
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("유튜브 링크 붙여넣기")
        btn_go = QPushButton("이동")
        btn_go.clicked.connect(self.load_url)
        url_layout.addWidget(self.url_input); url_layout.addWidget(btn_go); url_group.setLayout(url_layout)
        left_layout.addWidget(url_group)

        # 2. 설정
        setting_group = QGroupBox("캡처 설정")
        setting_layout = QVBoxLayout()
        h1 = QHBoxLayout(); h1.addWidget(QLabel("민감도(0.1~1.0):")); self.sensitivity_input = QLineEdit("0.9"); h1.addWidget(self.sensitivity_input); setting_layout.addLayout(h1)
        h2 = QHBoxLayout(); h2.addWidget(QLabel("시작 딜레이(초):")); self.delay_input = QLineEdit("3"); h2.addWidget(self.delay_input); setting_layout.addLayout(h2)
        setting_group.setLayout(setting_layout); left_layout.addWidget(setting_group)

        # 3. 유튜브 제어
        yt_control_group = QGroupBox("유튜브 제어")
        yt_layout = QVBoxLayout()
        self.btn_speed = QPushButton("⚡ 영상 2배속 적용"); self.btn_speed.clicked.connect(lambda: self.set_youtube_speed(2.0)); yt_layout.addWidget(self.btn_speed)
        self.btn_quality = QPushButton("ᴴᴰ 화질 최대 설정 (4K/HD)"); self.btn_quality.clicked.connect(self.set_youtube_max_quality); yt_layout.addWidget(self.btn_quality)
        yt_control_group.setLayout(yt_layout); left_layout.addWidget(yt_control_group)

        # 4. 캡처 제어 버튼 (원래 색상 그대로)
        btn_group = QGroupBox("캡처 제어")
        btn_layout = QVBoxLayout()
        self.btn_select = QPushButton("1. 영역 선택 모드 (영상 위 드래그)")
        self.btn_select.setStyleSheet("background-color: #d1ecf1; height: 35px;")
        self.btn_select.clicked.connect(self.toggle_selection_mode)
        
        self.btn_start = QPushButton("2. 캡처 시작")
        self.btn_start.setStyleSheet("background-color: #d4edda; height: 35px;")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_capture)
        
        self.btn_stop = QPushButton("캡처 중지")
        self.btn_stop.setStyleSheet("background-color: #f8d7da; height: 30px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_capture)

        self.btn_pdf = QPushButton("3. PDF 생성")
        self.btn_pdf.setStyleSheet("height: 35px; font-weight: bold;")
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self.create_pdf)

        btn_layout.addWidget(self.btn_select); btn_layout.addWidget(self.btn_start); btn_layout.addWidget(self.btn_stop); btn_layout.addWidget(self.btn_pdf)
        btn_group.setLayout(btn_layout); left_layout.addWidget(btn_group)

        # 5. 리스트 및 미리보기
        preview_group = QGroupBox("캡처 목록 및 확인")
        preview_layout = QVBoxLayout()
        self.list_widget = QListWidget(); self.list_widget.setMaximumHeight(150); self.list_widget.itemClicked.connect(self.show_image_preview); preview_layout.addWidget(self.list_widget)
        self.btn_delete = QPushButton("선택 항목 삭제"); self.btn_delete.clicked.connect(self.delete_selected_item); preview_layout.addWidget(self.btn_delete)
        self.image_preview_label = QLabel("리스트를 클릭하면 미리보기가 뜹니다."); self.image_preview_label.setAlignment(Qt.AlignCenter); self.image_preview_label.setMinimumHeight(200); self.image_preview_label.setStyleSheet("background-color: #e9ecef; border: 1px solid #ced4da;"); preview_layout.addWidget(self.image_preview_label)
        preview_group.setLayout(preview_layout); left_layout.addWidget(preview_group)

        # 6. 상태 표시줄
        self.status_label = QLabel("준비 완료"); self.status_label.setStyleSheet("color: blue; font-weight: bold;"); left_layout.addWidget(self.status_label)

        # --- 오른쪽 패널 (웹뷰 + 오버레이) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container); right_layout.setContentsMargins(0,0,0,0)
        self.webview = QWebEngineView(); self.webview.setUrl(QUrl("https://www.youtube.com")); right_layout.addWidget(self.webview)

        self.overlay = SelectionOverlay(self.webview)
        self.overlay.selection_finished.connect(self.finish_selection)
        
        splitter = QSplitter(Qt.Horizontal); splitter.addWidget(left_panel); splitter.addWidget(right_container); splitter.setStretchFactor(1, 3); main_layout.addWidget(splitter)

    # --- 기능 메서드 ---
    def load_url(self):
        url = self.url_input.text().strip()
        if url: self.webview.setUrl(QUrl(url if url.startswith("http") else "https://"+url))

    def set_youtube_speed(self, speed):
        self.webview.page().runJavaScript(f"document.querySelector('video').playbackRate = {speed};")
        self.status_label.setText(f"영상 속도를 {speed}배로 설정했습니다.")

    def set_youtube_max_quality(self):
        js = "var p = document.getElementById('movie_player'); if(p){var l=p.getAvailableQualityLevels(); p.setPlaybackQualityRange(l[0]);}"
        self.webview.page().runJavaScript(js)
        self.status_label.setText("최고 화질 적용을 시도했습니다.")

    def toggle_selection_mode(self):
        if not self.overlay.isVisible():
            self.overlay.resize(self.webview.size())
            self.overlay.show()
        
        is_active = not self.overlay.mode_active
        self.overlay.set_active(is_active)
        
        if is_active:
            self.btn_select.setText("선택 완료 (영상 조작 모드)")
            self.btn_select.setStyleSheet("background-color: #ffc107; height: 35px;")
            self.overlay.setFocus()
            self.status_label.setText("드래그하여 영역을 선택하고 방향키로 미세조절하세요.")
        else:
            self.btn_select.setText("1. 영역 선택 모드 다시 켜기")
            self.btn_select.setStyleSheet("background-color: #d1ecf1; height: 35px;")
            self.status_label.setText("영역이 고정되었습니다. 유튜브 영상을 재생하세요.")

    def finish_selection(self, area_dict):
        self.capture_area_dict = area_dict
        self.btn_start.setEnabled(True)
        self.status_label.setText(f"영역 설정됨: {area_dict['width']}x{area_dict['height']}")

    def start_capture(self):
        if not self.capture_area_dict:
            QMessageBox.warning(self, "오류", "영역을 먼저 선택하세요.")
            return

        self.btn_start.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pdf.setEnabled(False)
        
        self.captured_files = []
        self.list_widget.clear()
        self.image_preview_label.clear()
        self.last_captured_gray = None
        self.last_hash = None

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        delay_sec = int(self.delay_input.text())
        
        QTimer.singleShot(delay_sec * 1000, self.start_timed_capture)

        self.countdown_value = delay_sec
        self.update_countdown_label()

    def update_countdown_label(self):
        if self.capture_timer.isActive() or self.countdown_value < 0:
            return
        
        if self.countdown_value == 0:
            self.status_label.setText("캡처 감시 중... (중지 버튼으로 종료)")
        else:
            self.status_label.setText(f"{self.countdown_value}초 후 캡처 시작...")
        
        self.countdown_value -= 1
        if self.countdown_value >= 0:
            QTimer.singleShot(1000, self.update_countdown_label)
    
    def start_timed_capture(self):
        if self.btn_stop.isEnabled():
            self.capture_timer.start(1000)

    def stop_capture(self):
        self.capture_timer.stop()
        self.countdown_value = -1 
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pdf.setEnabled(len(self.captured_files) > 0)
        self.status_label.setText("캡처 중지됨.")

    def perform_capture(self):
        if not self.capture_area_dict:
            return
        try:
            w, h = self.capture_area_dict['width'], self.capture_area_dict['height']
            source_top_left_global = QPoint(self.capture_area_dict['left'], self.capture_area_dict['top'])
            source_top_left_local = self.webview.mapFromGlobal(source_top_left_global)
            source_rect = QRect(source_top_left_local, QSize(w, h))

            img = QImage(QSize(w, h), QImage.Format_ARGB32_Premultiplied)
            img.fill(Qt.transparent)

            self.overlay.hide()
            painter = QPainter(img)
            self.webview.render(painter, QPoint(), QRegion(source_rect))
            painter.end()
            self.overlay.show()

            ptr = img.constBits()
            ptr.setsize(img.sizeInBytes())
            arr = np.array(ptr).reshape(h, w, 4)
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            self.status_label.setText(f"캡처 오류: {e}")
            self.stop_capture()
            return

        should_save = False
        similarity_threshold = float(self.sensitivity_input.text())
        
        if self.last_captured_gray is None:
            should_save = True
            self.status_label.setText("첫 페이지 캡처!")
        else:
            score, _ = compare_ssim(self.last_captured_gray, img_gray, full=True)
            if score < similarity_threshold:
                self.status_label.setText(f"페이지 넘김 감지 (유사도: {score:.2f})")
                should_save = True

        if should_save:
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            curr_hash = imagehash.phash(pil_img)

            if self.last_hash is None or (curr_hash - self.last_hash > 5):
                filename = os.path.join(OUTPUT_FOLDER, f"score_{len(self.captured_files)+1:03d}.png")
                cv2.imwrite(filename, img_bgr)
                self.add_capture_item(filename)
                
                self.last_captured_gray = img_gray
                self.last_hash = curr_hash

    def add_capture_item(self, filepath):
        self.captured_files.append(filepath)
        self.list_widget.addItem(os.path.basename(filepath))
        self.list_widget.scrollToBottom()
        self.display_image(filepath)
        self.btn_pdf.setEnabled(True)

    def show_image_preview(self, item):
        path = os.path.join(OUTPUT_FOLDER, item.text())
        self.display_image(path)

    def display_image(self, filepath):
        if os.path.exists(filepath):
            pixmap = QPixmap(filepath)
            scaled = pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled)

    def delete_selected_item(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            item = self.list_widget.takeItem(row)
            if item is None: return
            
            full_path = os.path.join(OUTPUT_FOLDER, item.text())
            if full_path in self.captured_files: self.captured_files.remove(full_path)
            if os.path.exists(full_path): 
                try:
                    os.remove(full_path)
                except OSError as e:
                    self.status_label.setText(f"파일 삭제 오류: {e}")
            
            if self.list_widget.count() == 0:
                self.image_preview_label.clear()
                self.btn_pdf.setEnabled(False)

    def create_pdf(self):
        if not self.captured_files:
            QMessageBox.warning(self, "알림", "PDF로 만들 이미지가 없습니다.")
            return

        valid_files = [f for f in self.captured_files if os.path.exists(f)]
        if not valid_files:
            QMessageBox.warning(self, "알림", "캡처된 이미지 파일을 찾을 수 없습니다.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "PDF 저장", "악보.pdf", "PDF Files (*.pdf)")
        if path:
            try:
                self.status_label.setText("PDF 생성 중...")
                
                # 1. 이미지 객체 로드
                image_objects = [Image.open(f).convert("RGB") for f in valid_files]
                
                # 2. A4 비율 설정 (이미지 가로 크기 기준)
                base_width = image_objects[0].width
                a4_ratio = 297 / 210  # A4 세로/가로 비율
                page_height = int(base_width * a4_ratio)

                final_pages = []
                current_page = Image.new('RGB', (base_width, page_height), 'white')
                y_offset = 0

                # 3. 이미지 이어 붙이기 (페이지가 꽉 차면 다음 장으로)
                for img in image_objects:
                    # 너비 불일치 시 리사이징
                    if img.width != base_width:
                        new_h = int(img.height * (base_width / img.width))
                        img = img.resize((base_width, new_h), Image.Resampling.LANCZOS)

                    # 현재 페이지 남은 공간 확인
                    if y_offset + img.height > page_height:
                        final_pages.append(current_page)
                        current_page = Image.new('RGB', (base_width, page_height), 'white')
                        y_offset = 0
                    
                    current_page.paste(img, (0, y_offset))
                    y_offset += img.height
                
                final_pages.append(current_page) # 마지막 페이지 추가

                # 4. 페이지 번호 삽입 (하단 중앙)
                total_pages = len(final_pages)
                pages_with_numbers = []
                try:
                    # 시스템 폰트 시도 (arial), 실패 시 기본 폰트
                    font = ImageFont.truetype("arial.ttf", size=max(14, int(base_width/40)))
                except IOError:
                    font = ImageFont.load_default()

                for i, page in enumerate(final_pages, 1):
                    draw = ImageDraw.Draw(page)
                    page_num_text = f"{i} / {total_pages}"
                    
                    # 텍스트 크기 계산하여 중앙 배치
                    bbox = draw.textbbox((0, 0), page_num_text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    
                    x = (base_width - text_w) / 2
                    y = page_height - text_h - 10 # 아래쪽 여백 10px
                    
                    draw.text((x, y), page_num_text, font=font, fill="black")
                    pages_with_numbers.append(page)

                # 5. 저장
                if pages_with_numbers:
                    pages_with_numbers[0].save(path, save_all=True, append_images=pages_with_numbers[1:])
                    QMessageBox.information(self, "성공", f"PDF가 저장되었습니다! (총 {total_pages}페이지)")
                    
                    # 완료 후 목록 초기화 (사용자 선택)
                    self.captured_files = []
                    self.list_widget.clear()
                    self.image_preview_label.clear()
                    self.btn_pdf.setEnabled(False)
                    self.status_label.setText("PDF 저장 완료.")
                
            except Exception as e:
                QMessageBox.critical(self, "PDF 저장 오류", f"PDF를 저장하는 중 오류가 발생했습니다:\n{e}")

    def resizeEvent(self, event):
        if hasattr(self, 'overlay') and self.overlay.isVisible(): 
            self.overlay.resize(self.webview.size())
        super().resizeEvent(event)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv); window = MainWindow(); window.show(); sys.exit(app.exec_())