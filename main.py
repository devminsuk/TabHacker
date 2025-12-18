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
                             QSplitter, QFrame)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QRect, QRectF, QSize, QPoint, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QPainterPath, QRegion

from skimage.metrics import structural_similarity as compare_ssim

# --- 설정 ---
OUTPUT_FOLDER = "captured_scores"

class SelectionOverlay(QWidget):
    """개선된 영역 선택 오버레이: 조작 차단(Lock) 기능 추가"""
    selection_finished = pyqtSignal(dict) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.start_pos = None
        self.current_pos = None
        self.is_selecting = False
        self.mode_active = False   # 영역 선택 드래그 모드
        self.is_locked = False     # 캡처 중 유튜브 조작 차단 모드
        self.confirmed_rect = None
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.hide()

    def set_active(self, active):
        """영역 선택 모드 활성화/비활성화"""
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
        """캡처 중 유튜브 조작 차단 설정"""
        self.is_locked = lock
        self.mode_active = False
        if lock:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.setCursor(Qt.ForbiddenCursor) # 금지 표시 커서
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
            color_alpha = 160 if self.mode_active else 50
            overlay_color = QColor(0, 0, 0, color_alpha)
            path = QPainterPath()
            path.addRect(QRectF(self.rect()))
            if rect:
                path.addRect(QRectF(rect))
            painter.fillPath(path, overlay_color)

        if rect:
            if self.is_locked:
                color = QColor(255, 0, 0)
                text = "⚠️ 캡처 중 (조작 차단됨)"
            elif self.mode_active:
                color = QColor(0, 174, 255)
                text = f"{rect.width()} x {rect.height()}"
            else:
                color = QColor(0, 255, 0)
                text = None

            painter.setPen(QPen(color, 2, Qt.SolidLine))
            painter.drawRect(rect)
            if text:
                painter.setPen(Qt.white)
                painter.drawText(rect.topLeft() + QPoint(5, -10), text)
        elif self.mode_active:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "마우스로 드래그하여 영역을 선택하세요.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube 악보 캡처 Pro (자동 재생 지원)")
        self.resize(1300, 850)
        self.capture_area_dict = None
        self.captured_files = []

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
        
        # --- 왼쪽 패널 ---
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

        # 3. 캡처 제어 버튼
        btn_group = QGroupBox("캡처 제어")
        btn_layout = QVBoxLayout()
        self.btn_select = QPushButton("1. 영역 선택 모드")
        self.btn_select.setStyleSheet("background-color: #d1ecf1; height: 35px;")
        self.btn_select.clicked.connect(self.toggle_selection_mode)
        
        self.btn_start = QPushButton("2. 캡처 시작 (자동 재생)")
        self.btn_start.setStyleSheet("background-color: #d4edda; height: 35px; font-weight: bold;")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_capture)
        
        self.btn_stop = QPushButton("캡처 중지 (일시정지)")
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
        self.image_preview_label = QLabel("미리보기"); self.image_preview_label.setAlignment(Qt.AlignCenter); self.image_preview_label.setMinimumHeight(200); self.image_preview_label.setStyleSheet("background-color: #e9ecef; border: 1px solid #ced4da;"); preview_layout.addWidget(self.image_preview_label)
        preview_group.setLayout(preview_layout); left_layout.addWidget(preview_group)

        # 6. 상태 표시줄
        self.status_label = QLabel("준비 완료"); self.status_label.setStyleSheet("color: blue; font-weight: bold;"); left_layout.addWidget(self.status_label)

        # --- 오른쪽 패널 (웹뷰) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container); right_layout.setContentsMargins(0,0,0,0)
        self.webview = QWebEngineView(); self.webview.setUrl(QUrl("https://www.youtube.com")); right_layout.addWidget(self.webview)

        self.overlay = SelectionOverlay(self.webview)
        self.overlay.selection_finished.connect(self.finish_selection)
        
        splitter = QSplitter(Qt.Horizontal); splitter.addWidget(left_panel); splitter.addWidget(right_container); splitter.setStretchFactor(1, 3); main_layout.addWidget(splitter)

    # --- 유튜브 제어 ---
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
        if url: self.webview.setUrl(QUrl(url if url.startswith("http") else "https://"+url))

    def toggle_selection_mode(self):
        if self.overlay.isVisible() and self.overlay.mode_active:
            self.overlay.set_active(False)
            self.overlay.hide()
            self.btn_select.setText("1. 영역 선택 모드")
        else:
            self.overlay.resize(self.webview.size())
            self.overlay.set_active(True)
            self.btn_select.setText("선택 취소")
            self.overlay.setFocus()

    def finish_selection(self, area_dict):
        self.capture_area_dict = area_dict
        self.btn_start.setEnabled(True)
        self.btn_select.setText("1. 영역 선택 모드")
        self.status_label.setText(f"영역 설정됨: {area_dict['width']}x{area_dict['height']}")
        QMessageBox.information(self, "완료", "영역이 설정되었습니다. '2. 캡처 시작'을 누르세요.")

    def start_capture(self):
        self.btn_start.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pdf.setEnabled(False)
        
        self.overlay.set_lock(True) # 조작 차단
        self.overlay.show()

        self.captured_files = []
        self.list_widget.clear()
        self.image_preview_label.clear() # 미리보기 초기화
        self.last_captured_gray = None
        self.last_hash = None
        if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

        self.countdown_value = int(self.delay_input.text())
        self.run_countdown()

    def run_countdown(self):
        if self.countdown_value > 0:
            self.status_label.setText(f"{self.countdown_value}초 후 시작 및 자동 재생...")
            self.countdown_value -= 1
            QTimer.singleShot(1000, self.run_countdown)
        else:
            self.status_label.setText("캡처 진행 중... (유튜브 조작 불가)")
            self.set_youtube_state("quality")
            self.set_youtube_state("speed", 2.0)
            self.set_youtube_state("play") # 자동 재생
            self.capture_timer.start(1000)

    def stop_capture(self):
        self.capture_timer.stop()
        self.set_youtube_state("pause") # 자동 일시정지
        self.overlay.set_lock(False)    # 조작 차단 해제
        
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pdf.setEnabled(len(self.captured_files) > 0)
        self.status_label.setText("캡처 중지됨.")

    def perform_capture(self):
        if not self.capture_area_dict: return
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
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            should_save = False
            threshold = float(self.sensitivity_input.text())
            
            if self.last_captured_gray is None:
                should_save = True
            else:
                score, _ = compare_ssim(self.last_captured_gray, img_gray, full=True)
                if score < threshold: should_save = True

            if should_save:
                pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                curr_hash = imagehash.phash(pil_img)
                if self.last_hash is None or (curr_hash - self.last_hash > 5):
                    filename = os.path.join(OUTPUT_FOLDER, f"score_{len(self.captured_files)+1:03d}.png")
                    cv2.imwrite(filename, img_bgr)
                    self.captured_files.append(filename)
                    self.list_widget.addItem(os.path.basename(filename))
                    self.list_widget.scrollToBottom()
                    
                    # --- 미리보기 갱신 ---
                    self.display_image(filename)
                    self.btn_pdf.setEnabled(True)
                    # -------------------

                    self.last_captured_gray = img_gray
                    self.last_hash = curr_hash
        except Exception as e:
            print(f"Capture Error: {e}")

    def show_image_preview(self, item):
        path = os.path.join(OUTPUT_FOLDER, item.text())
        self.display_image(path)

    def display_image(self, filepath):
        if os.path.exists(filepath):
            pixmap = QPixmap(filepath)
            self.image_preview_label.setPixmap(pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def delete_selected_item(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            item = self.list_widget.takeItem(row)
            full_path = os.path.join(OUTPUT_FOLDER, item.text())
            if full_path in self.captured_files: self.captured_files.remove(full_path)
            if os.path.exists(full_path): os.remove(full_path)
            
            if self.list_widget.count() == 0: 
                self.image_preview_label.clear()
                self.btn_pdf.setEnabled(False)
            else:
                # 삭제 후 마지막 항목 미리보기
                last_item = self.list_widget.item(self.list_widget.count() - 1)
                self.show_image_preview(last_item)

    def create_pdf(self):
        if not self.captured_files: return
        path, _ = QFileDialog.getSaveFileName(self, "PDF 저장", "악보_결과.pdf", "PDF Files (*.pdf)")
        if not path: return
        
        try:
            image_objects = [Image.open(f).convert("RGB") for f in self.captured_files if os.path.exists(f)]
            # A4 비율 자동 맞춤 생성
            base_width = image_objects[0].width
            page_height = int(base_width * (297 / 210))
            final_pages = []
            current_page = Image.new('RGB', (base_width, page_height), 'white')
            y_offset = 0

            for img in image_objects:
                if img.width != base_width:
                    new_h = int(img.height * (base_width / img.width))
                    img = img.resize((base_width, new_h), Image.Resampling.LANCZOS)
                if y_offset + img.height > page_height:
                    final_pages.append(current_page)
                    current_page = Image.new('RGB', (base_width, page_height), 'white')
                    y_offset = 0
                current_page.paste(img, (0, y_offset))
                y_offset += img.height
            final_pages.append(current_page)

            # 페이지 번호 삽입
            try:
                draw_font = ImageFont.truetype("arial.ttf", size=max(14, int(base_width/40)))
            except IOError:
                draw_font = ImageFont.load_default()

            for i, page in enumerate(final_pages, 1):
                draw = ImageDraw.Draw(page)
                text = f"{i} / {len(final_pages)}"
                
                # 텍스트 크기 계산 (Pillow 버전에 따라 다름)
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((0, 0), text, font=draw_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                else:
                    text_w, text_h = draw.textsize(text, font=draw_font)

                draw.text(((base_width - text_w) // 2, page_height - text_h - 20), text, fill="black", font=draw_font)

            final_pages[0].save(path, save_all=True, append_images=final_pages[1:])
            QMessageBox.information(self, "성공", f"PDF 저장 완료! (총 {len(final_pages)}페이지)")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"PDF 생성 실패: {e}")

    def resizeEvent(self, event):
        if hasattr(self, 'overlay'): self.overlay.resize(self.webview.size())
        super().resizeEvent(event)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
