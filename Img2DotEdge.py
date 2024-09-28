import time
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2  # OpenCV for edge detection
import numpy as np
import threading

class PixelArtApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("ドット絵変換ツール（特徴検出あり）")
        self.geometry("900x600")  # ウィンドウサイズを大きめに設定

        # 左側：画像のドラッグ＆ドロップ領域（枠付きのLabel）
        self.left_frame = tk.Frame(self, width=300, height=300)
        self.left_frame.pack(side="left", padx=10, pady=10)
        
        # 画像表示枠を追加（初期サイズ固定）
        self.label_width = 300  # 初期表示のラベルサイズ（幅）
        self.label_height = 300  # 初期表示のラベルサイズ（高さ）
        self.original_image_label = tk.Label(self.left_frame, text="ここに画像をドロップ", width=self.label_width, height=self.label_height,
                                             relief="solid", bd=2, anchor="center")
        self.original_image_label.pack(pady=10)

        # ドラッグ＆ドロップ領域設定
        self.original_image_label.drop_target_register(DND_FILES)
        self.original_image_label.dnd_bind('<<Drop>>', self.on_drop)

        # 中央：ピクセルサイズ、色数、エッジ検出パラメータの入力スライダーとボタン
        self.center_frame = tk.Frame(self, width=200, height=400)
        self.center_frame.pack(side="left", padx=10, pady=10)

        # ピクセルサイズのスライダー
        tk.Label(self.center_frame, text="ピクセルサイズ (1〜50):").pack(pady=5)
        self.pixel_size_scale = tk.Scale(self.center_frame, from_=1, to=50, orient=tk.HORIZONTAL)
        self.pixel_size_scale.set(16)
        self.pixel_size_scale.pack(pady=5)

        # 色数のスライダー
        tk.Label(self.center_frame, text="色数 (2〜256):").pack(pady=5)
        self.num_colors_scale = tk.Scale(self.center_frame, from_=2, to=256, orient=tk.HORIZONTAL)
        self.num_colors_scale.set(16)
        self.num_colors_scale.pack(pady=5)

        # エッジ検出パラメータの設定
        tk.Label(self.center_frame, text="エッジ検出の閾値1 (低)").pack(pady=5)
        self.edge_thresh1_scale = tk.Scale(self.center_frame, from_=50, to=300, orient=tk.HORIZONTAL)
        self.edge_thresh1_scale.set(100)  # デフォルト値
        self.edge_thresh1_scale.pack(pady=5)

        tk.Label(self.center_frame, text="エッジ検出の閾値2 (高)").pack(pady=5)
        self.edge_thresh2_scale = tk.Scale(self.center_frame, from_=100, to=500, orient=tk.HORIZONTAL)
        self.edge_thresh2_scale.set(200)  # デフォルト値
        self.edge_thresh2_scale.pack(pady=5)

        tk.Label(self.center_frame, text="ガウスぼかしのカーネルサイズ").pack(pady=5)
        self.blur_kernel_scale = tk.Scale(self.center_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.blur_kernel_scale.set(5)  # デフォルトは5（適度なぼかし）
        self.blur_kernel_scale.pack(pady=5)

        # ドット変換ボタン
        self.convert_button = tk.Button(self.center_frame, text="ドット絵に変換", command=self.run_conversion)
        self.convert_button.pack(pady=10)

        # クリアボタン
        self.clear_button = tk.Button(self.center_frame, text="クリア", command=self.clear_screen)
        self.clear_button.pack(pady=10)

        # 右側：変換されたドット絵と保存ボタン
        self.right_frame = tk.Frame(self, width=300, height=300)
        self.right_frame.pack(side="right", padx=10, pady=10)
        
        self.pixel_art_label = tk.Label(self.right_frame, text="ここに変換後の画像を表示", width=self.label_width, height=self.label_height,
                                        relief="solid", bd=2, anchor="center")
        self.pixel_art_label.pack(pady=10)
        
        self.save_button = tk.Button(self.right_frame, text="ドット絵を保存", command=self.save_pixel_art)
        self.save_button.pack(pady=10)
        
        self.original_image = None
        self.pixel_art_image = None
        self.splash_window = None
        self.progress_var = None  # プログレスバーの値
        self.progress_bar = None  # プログレスバーウィジェット
        self.image_tk = None

    def on_drop(self, event):
        # ドラッグ＆ドロップされたファイルを取得
        file_path = event.data.strip('{}')

        # 対応ファイルかチェック
        if os.path.splitext(file_path)[1].lower() in ['.png', '.jpg', '.jpeg']:
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.original_image_label)
        else:
            messagebox.showerror("エラー", "PNGまたはJPEG画像をドロップしてください")

    def display_image(self, image, label):
        # ラベルのサイズに合わせたリサイズを行う（中央部分ではなく全体を表示する）
        img_width, img_height = image.size
        label_width = self.label_width
        label_height = self.label_height

        # アスペクト比を維持しつつ、表示領域に収まるようにリサイズ
        if img_width > label_width or img_height > label_height:
            ratio = min(label_width / img_width, label_height / img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            image_resized = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            image_resized = image.copy()

        # 画像をラベルの中央に表示するようにする
        self.image_tk = ImageTk.PhotoImage(image_resized)
        label.configure(image=self.image_tk, text='')  # テキストをクリアして画像を表示
        label.image = self.image_tk

    def clear_screen(self):
        """画面を初期状態に戻す処理"""
        # 左側のラベルを初期状態に戻す
        self.original_image_label.configure(image='', text="ここに画像をドロップ")
        self.original_image_label.image = None
        
        # 右側のラベルを初期状態に戻す
        self.pixel_art_label.configure(image='', text="ここに変換後の画像を表示")
        self.pixel_art_label.image = None

        # 内部で保持している画像データもリセット
        self.original_image = None
        self.pixel_art_image = None
        self.image_tk = None

    def run_conversion(self):
        conversion_thread = threading.Thread(target=self.convert_to_pixel_art)
        conversion_thread.start()

    def show_splash(self):
        """スプラッシュウィンドウをメイン画面の中央に表示"""
        self.splash_window = tk.Toplevel(self)
        self.splash_window.title("変換中...")

        main_window_x = self.winfo_x()
        main_window_y = self.winfo_y()
        main_window_width = self.winfo_width()
        main_window_height = self.winfo_height()
        splash_width = 300
        splash_height = 150

        splash_x = main_window_x + (main_window_width - splash_width) // 2
        splash_y = main_window_y + (main_window_height - splash_height) // 2
        self.splash_window.geometry(f"{splash_width}x{splash_height}+{splash_x}+{splash_y}")

        # プログレスバーとメッセージを表示
        label = tk.Label(self.splash_window, text="ドット絵に変換中です...")
        label.pack(pady=10)
        
        self.progress_var = tk.DoubleVar()  # プログレスバーの値
        self.progress_bar = ttk.Progressbar(self.splash_window, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        
        self.splash_window.grab_set()

    def hide_splash(self):
        if self.splash_window:
            self.splash_window.destroy()
            self.splash_window = None

    def convert_to_pixel_art(self):
        if not self.original_image:
            messagebox.showerror("エラー", "画像を選択してください")
            return

        self.show_splash()

        try:
            pixel_size = self.pixel_size_scale.get()
            num_colors = self.num_colors_scale.get()
            edge_thresh1 = self.edge_thresh1_scale.get()
            edge_thresh2 = self.edge_thresh2_scale.get()
            blur_kernel = self.blur_kernel_scale.get()

            # 進捗の設定
            steps = 10  # テスト用に10ステップと仮定
            for i in range(steps):
                # 仮の処理時間
                time.sleep(0.3)  # 各ステップの処理に0.3秒かかると仮定
                
                # 進捗を更新
                progress_value = (i + 1) * (100 / steps)
                self.progress_var.set(progress_value)
                self.splash_window.update_idletasks()

            # ドット絵変換処理
            original_cv_image = np.array(self.original_image.convert("RGB"))
            original_cv_image = cv2.cvtColor(original_cv_image, cv2.COLOR_RGB2BGR)

            # ガウスぼかしの適用
            if blur_kernel % 2 == 0:
                blur_kernel += 1  # カーネルサイズは奇数である必要がある
            blurred_image = cv2.GaussianBlur(original_cv_image, (blur_kernel, blur_kernel), 0)

            # Cannyエッジ検出の実行
            edges = cv2.Canny(blurred_image, edge_thresh1, edge_thresh2)

            # 残りの処理（エッジを適用した画像を元にドット絵に変換）
            self.pixel_art_image = self.image_to_pixel_art(blurred_image, pixel_size=pixel_size, num_colors=num_colors)
            self.pixel_art_image_pil = Image.fromarray(cv2.cvtColor(self.pixel_art_image, cv2.COLOR_BGR2RGB))
            self.display_image(self.pixel_art_image_pil, self.pixel_art_label)
        
        finally:
            self.hide_splash()

    def save_pixel_art(self):
        if self.pixel_art_image_pil is None:
            messagebox.showerror("エラー", "変換されたドット絵がありません")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp")]
        )
        if file_path:
            self.pixel_art_image_pil.save(file_path)
            messagebox.showinfo("保存完了", "ドット絵を保存しました")

    def image_to_pixel_art(self, image, pixel_size=16, num_colors=16):
        edges = cv2.Canny(image, 100, 200)

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[np.where((edges_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

        img_small = cv2.resize(image, (image.shape[1] // pixel_size, image.shape[0] // pixel_size), interpolation=cv2.INTER_NEAREST)
        
        Z = img_small.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        K = num_colors
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        img_small = centers[labels.flatten()].reshape((img_small.shape))

        pixel_art = cv2.resize(img_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        pixel_art = cv2.addWeighted(pixel_art, 1, edges_colored, 0.5, 0)

        return pixel_art

app = PixelArtApp()
app.mainloop()
