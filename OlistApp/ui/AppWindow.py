"""
Cửa sổ chính sau khi đăng nhập: sidebar bên trái + 4 màn hình.

  6.1.1 Tổng quan hệ thống            (+ 6.2.1 các mô hình đã tích hợp)
  6.1.2 Dự đoán giao hàng trễ         (+ 6.2.2 kết quả mô hình XGBoost)
  6.1.3 Dự đoán đánh giá tiêu cực     (+ 6.2.3 kết quả mô hình Random Forest, nhánh sau giao)
  6.1.4 Phân nhóm nhà bán hàng        (+ 6.2.4 kết quả K-Means, K = 2)

Mô hình tích hợp theo mục 5.5.2 của báo cáo; phần tính toán nằm ở ml/predictor.py và ml/seller_clustering.py.
"""
import html
import textwrap

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QStackedWidget, QTabWidget,
                             QVBoxLayout, QWidget)
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from ml import seller_clustering as sc
from ml.paths import RESULT_DIR
from ml.predictor import OrderPredictor
from ml.display import STATE_NAMES, state_label, search_text
from ml.seller_selection import rank_sellers, CRITERIA
from ui import theme as th

APP_NAME = "Smart E-commerce Order Management"
MODEL_VI = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
            "random_forest": "Random Forest", "xgboost": "XGBoost"}
FILES = {"delivery": "delivery", "review": "review_post"}       # tiền tố tệp trong results/
TASKS = {
    "delivery": {"event": "giao hàng trễ", "positive": "Cần kiểm tra giao hàng", "negative": "Theo dõi thông thường",
                 "actual": {1: "giao trễ", 0: "giao đúng hẹn hoặc sớm"},
                 "labels": ["Đúng hẹn (0)", "Giao trễ (1)"],
                 "when": "Tính khi đơn được xác nhận."},
    "review": {"event": "đánh giá 1–2 sao", "positive": "Nguy cơ đánh giá thấp", "negative": "Ít nguy cơ",
               "actual": {1: "khách đánh giá 1–2 sao", 0: "khách đánh giá 3–5 sao"},
               "labels": ["3–5 sao (0)", "1–2 sao (1)"],
               "when": "Tính sau khi giao thành công."},
}
PAGES = [("Việc cần theo dõi", "Tra cứu và ưu tiên kiểm tra trên dữ liệu lịch sử Olist", th.ICON_HOME),
         ("Theo dõi giao hàng", "Kiểm tra cảnh báo, xem thông tin đối chiếu và hướng xử lý", th.ICON_CLOCK),
         ("Chăm sóc sau giao", "Ưu tiên kiểm tra trải nghiệm khách hàng sau khi giao thành công", th.ICON_STAR),
         ("Hiệu quả nhà bán", "So sánh lịch sử và đề xuất theo tiêu chí bạn chọn", th.ICON_PEOPLE)]
NAVY_CMAP = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", th.NAVY])
DIVERGING = LinearSegmentedColormap.from_list("navy_orange", [th.NAVY, "#FFFFFF", th.ORANGE])
FEATURE_VI = {f: sc.FEATURES[f][0] for f in sc.FEATURE_COLS}


def make_canvas(min_height=300):
    fig = Figure(facecolor="white", layout="constrained")
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(min_height)
    return fig, canvas


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8C2D1")
    ax.tick_params(colors=th.MUTED, labelsize=9)
    ax.grid(alpha=0.25, color="#B8C2D1")
    ax.set_axisbelow(True)
    style_title(ax)
    ax.xaxis.label.set_color(th.TEXT)
    ax.yaxis.label.set_color(th.TEXT)


def style_title(ax):
    ax.title.set_color(th.NAVY)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")


def test_metrics(df):
    y, s, p = df["target"], df["score"], df["prediction"]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"ap": average_precision_score(y, s), "roc": roc_auc_score(y, s), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "base": y.mean(), "flagged": p.mean(), "n": len(df), "cm": np.array([[tn, fp], [fn, tp]])}


class AppWindow(QMainWindow):
    loggedOut = pyqtSignal()

    def __init__(self, username="admin", role="Quản trị viên"):
        super().__init__()
        self.username, self.role = username, role
        self.rng = np.random.default_rng(42)
        self.order_id = None
        self.predictor = OrderPredictor()
        self.pred = {k: pd.read_csv(RESULT_DIR / f"{FILES[k]}_test_predictions.csv") for k in TASKS}
        self.metrics = {k: test_metrics(df) for k, df in self.pred.items()}
        self.chosen = {"delivery": self.predictor.delivery_name, "review": self.predictor.review_name}
        self.threshold = {"delivery": self.predictor.delivery_threshold, "review": self.predictor.review_threshold}
        self.k_report = pd.read_csv(RESULT_DIR / "kmeans_k_selection.csv")
        self.stability = pd.read_csv(RESULT_DIR / "kmeans_stability.csv")
        self.ui = {"delivery": {}, "review": {}}
        self.orderEdits = []
        self.cluster = None

        self.setWindowTitle(APP_NAME)
        self.resize(1400, 860)
        self.setMinimumSize(1180, 720)
        self.setStyleSheet(th.QSS)
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._sidebar())
        lay.addWidget(self._content(), 1)

        self.cluster = sc.fit(self.predictor.kmeans["k"], self.predictor.kmeans["model"])
        self.sellerHistory = sc.build_seller_data(include_all=True)[0]
        self.refreshSellerRanking()
        self.refreshOverview()
        for key in TASKS:
            self.showModelResults(key)
        self.goto(0)

    # ================================================================ khung
    def _sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(250)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 24, 0, 18)
        lay.setSpacing(2)
        head = QVBoxLayout()
        head.setContentsMargins(22, 0, 16, 0)
        head.setSpacing(4)
        bar = QFrame()
        bar.setFixedSize(36, 4)
        bar.setStyleSheet(f"background: {th.ORANGE}; border-radius: 2px;")
        brand = QLabel("Smart E-commerce\nOrder Management")
        brand.setObjectName("brand")
        sub = QLabel("Hệ thống quản lý đơn hàng")
        sub.setObjectName("brandSub")
        for w in (bar, brand, sub):
            head.addWidget(w)
        lay.addLayout(head)
        lay.addSpacing(26)
        cap = QLabel("CHỨC NĂNG")
        cap.setObjectName("navCaption")
        cap.setContentsMargins(22, 0, 0, 6)
        lay.addWidget(cap)

        self.navGroup = QButtonGroup(self)
        for i, (title, _, icon) in enumerate(PAGES):
            btn = QPushButton("  " + title)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.setIcon(th.glyph_icon(icon, "#FFFFFF"))
            btn.setIconSize(QSize(18, 18))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(46)
            self.navGroup.addButton(btn, i)
            lay.addWidget(btn)
        self.navGroup.idClicked.connect(self.goto)
        lay.addStretch(1)

        foot = QVBoxLayout()
        foot.setContentsMargins(22, 0, 22, 0)
        foot.setSpacing(2)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #2A426B;")
        foot.addWidget(line)
        foot.addSpacing(12)
        name = QLabel(self.username)
        name.setObjectName("userName")
        role = QLabel(self.role)
        role.setObjectName("userRole")
        foot.addWidget(name)
        foot.addWidget(role)
        foot.addSpacing(10)
        self.pushButtonLogout = QPushButton("  Đăng xuất")
        self.pushButtonLogout.setObjectName("logout")
        self.pushButtonLogout.setIcon(th.glyph_icon(th.ICON_POWER, "#FFFFFF"))
        self.pushButtonLogout.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pushButtonLogout.clicked.connect(self.logout)
        foot.addWidget(self.pushButtonLogout)
        lay.addLayout(foot)
        return side

    def _content(self):
        content = QWidget()
        content.setObjectName("content")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = QFrame()
        top.setObjectName("topbar")
        top.setFixedHeight(72)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(24, 10, 24, 10)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.labelPageTitle = QLabel()
        self.labelPageTitle.setObjectName("pageTitle")
        self.labelPageSub = QLabel()
        self.labelPageSub.setObjectName("pageSub")
        titles.addWidget(self.labelPageTitle)
        titles.addWidget(self.labelPageSub)
        tl.addLayout(titles, 1)
        lay.addWidget(top)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_overview())
        self.stack.addWidget(self._page_model("delivery"))
        self.stack.addWidget(self._page_model("review"))
        self.stack.addWidget(self._page_sellers())
        lay.addWidget(self.stack, 1)
        return content

    def _page(self):
        page = QWidget()
        page.setObjectName("page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)
        return page, lay

    def _sample_orders(self, n=300):
        """Danh sách (mã đơn, cụm) cho ô xổ xuống: lấy ngẫu nhiên n đơn của tập kiểm thử có đủ đặc trưng cho
        cả hai mô hình, chia đều theo cụm của nhà bán (chỉ xét đơn có một nhà bán đã được phân nhóm)."""
        if not hasattr(self, "_orders"):
            p = self.predictor
            ids = p.delivery_store.labels["test"]["order_id"]
            ids = ids[ids.isin(p.review_store.index.index)]
            pairs = p.items[p.items["order_id"].isin(ids)][["order_id", "seller_id"]].drop_duplicates()
            pairs = pairs[~pairs["order_id"].duplicated(keep=False)]              # đơn chỉ có một nhà bán
            pairs["cluster"] = pairs["seller_id"].map(p.clusters["cluster"])
            pairs = pairs.dropna(subset="cluster")
            # mức rủi ro theo dự đoán của hai mô hình trên tập kiểm thử
            late = pairs["order_id"].map(self.pred["delivery"].set_index("order_id")["prediction"])
            low = pairs["order_id"].map(self.pred["review"].set_index("order_id")["prediction"])
            pairs = pairs[late.notna() & low.notna()].copy()
            pairs["risk"] = np.select(
                [late.eq(1) & low.eq(1), late.eq(1), low.eq(1)],
                ["nguy cơ cao cả hai", "nguy cơ giao trễ", "nguy cơ đánh giá thấp"], "bình thường")
            codes, risks = sorted(pairs["cluster"].unique()), sorted(pairs["risk"].unique())
            picked = []
            for code in codes:
                in_cluster = pairs[pairs["cluster"] == code]
                quota, chosen = n // len(codes), []
                for risk in risks:                         # chia đều cho từng mức rủi ro trong cụm
                    pool = in_cluster.loc[in_cluster["risk"] == risk, "order_id"].to_numpy()
                    chosen += self.rng.choice(pool, size=min(quota // len(risks), len(pool)), replace=False).tolist()
                rest = in_cluster.loc[~in_cluster["order_id"].isin(chosen), "order_id"].to_numpy()
                chosen += self.rng.choice(rest, size=min(quota - len(chosen), len(rest)), replace=False).tolist()
                tag = in_cluster.set_index("order_id")["risk"]
                picked += [(str(o), f"{code}   ·   {tag[o]}") for o in chosen]
            self._orders = sorted(picked)
            self._order_risk = pairs.set_index("order_id")["risk"].reindex([o for o, _ in picked]).value_counts()
        return self._orders

    def _order_input(self):
        """Ô xổ xuống chọn mã đơn (vẫn gõ được mã khác) + nút Dự đoán, dùng chung cho các màn hình."""
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.setMaxVisibleItems(12)
        for order_id, _ in self._sample_orders():
            combo.addItem(self.predictor.order_codes.display(order_id))
            combo.setItemData(combo.count() - 1, order_id, Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(-1)
        combo.lineEdit().setPlaceholderText("Chọn hoặc nhập mã ORD-000001 / mã gốc")
        combo.lineEdit().addAction(th.glyph_icon(th.ICON_SEARCH, th.MUTED),
                                   QLineEdit.ActionPosition.LeadingPosition)
        predict = QPushButton("Xem cảnh báo")
        predict.setObjectName("primary")
        predict.setCursor(Qt.CursorShape.PointingHandCursor)
        predict.clicked.connect(lambda: self.lookup(combo.currentText()))
        combo.lineEdit().returnPressed.connect(lambda: self.lookup(combo.currentText()))
        self.orderEdits.append(combo)
        return combo, predict

    def _order_count(self):
        label = QLabel(f"Danh sách mẫu {len(self._sample_orders()):,} / {len(self.predictor.order_ids()):,} đơn, chọn ngẫu nhiên.")
        label.setWordWrap(True)
        label.setObjectName("muted")
        return label

    def _empty(self, text):
        label = QLabel(text)
        label.setObjectName("empty")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        return label

    def goto(self, index):
        self.stack.setCurrentIndex(index)
        self.navGroup.button(index).setChecked(True)
        self.labelPageTitle.setText(PAGES[index][0])
        self.labelPageSub.setText(PAGES[index][1])

    def logout(self):
        self.loggedOut.emit()
        self.close()

    # ================================================================ 6.1.1 Tổng quan
    def _priority_table(self, key):
        df = self.pred[key]
        flagged = df[df["prediction"].eq(1)].sort_values("score", ascending=False)
        table = th.make_table()
        table.setSelectionMode(table.SelectionMode.SingleSelection)
        show = pd.DataFrame({"Mã đơn": flagged["order_id"].map(self.predictor.order_codes.display),
                             "Điểm cảnh báo /100": (flagged["score"] * 100).round(1)})
        show.attrs["raw_ids"] = {"Mã đơn": flagged["order_id"].tolist()}
        th.fill_table(table, show, decimals=1)
        table.setColumnWidth(0, 155)
        def open_order(row, column):
            self.lookup(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            self.goto(1 if key == "delivery" else 2)
        table.cellDoubleClicked.connect(open_order)
        return table

    def _guidance(self, key, result, order_id):
        if result is None:
            return "Chưa đủ dữ liệu để chấm điểm. Kiểm tra trạng thái và thông tin đơn trước khi xử lý."
        flagged = result["label"] == 1
        if flagged:
            title = "Ưu tiên kiểm tra tiến độ giao hàng" if key == "delivery" else "Ưu tiên chăm sóc khách sau giao"
            action = ("Liên hệ nhà bán xác nhận tiến độ, kiểm tra bàn giao vận chuyển và cập nhật cho khách."
                      if key == "delivery" else
                      "Hỏi thăm tình trạng sản phẩm và trải nghiệm nhận hàng; hỗ trợ khi khách phản ánh vấn đề.")
        else:
            title = "Theo dõi thông thường — điểm dưới ngưỡng cảnh báo"
            action = "Tiếp tục theo dõi đơn và phản hồi của khách; vẫn kiểm tra nếu có vấn đề phát sinh."
        o = self.predictor.orders.loc[order_id]
        facts = [f"Ngày hẹn giao: {str(o['order_estimated_delivery_date'])[:10]}."]
        if key == "review":
            actual = pd.to_datetime(o["order_delivered_customer_date"], errors="coerce")
            expected = pd.to_datetime(o["order_estimated_delivery_date"], errors="coerce")
            if pd.notna(actual) and pd.notna(expected):
                days = (actual.normalize() - expected.normalize()).days
                facts.append(f"Thực tế giao trễ {days} ngày." if days > 0 else "Thực tế giao đúng hẹn hoặc sớm.")
        ids = self.predictor.items.loc[self.predictor.items["order_id"].eq(order_id), "seller_id"].unique()
        for _, s in self.sellerHistory[self.sellerHistory["seller_id"].isin(ids)].iterrows():
            if s["completed_order_count"] > 0 and pd.notna(s["late_order_rate"]) and pd.notna(s["mean_delivery_days"]):
                facts.append(f"Nhà bán {self.predictor.seller_codes.display(s['seller_id'])}: "
                             f"{int(s['completed_order_count'])} đơn lịch sử, tỷ lệ trễ {s['late_order_rate']:.1%}, "
                             f"giao trung bình {s['mean_delivery_days']:.1f} ngày (toàn kỳ).")
        items = "".join(f"<li>{html.escape(t)}</li>" for t in facts)
        return (f"<div style='font-size:15px; font-weight:600; color:{th.NAVY}'>{html.escape(title)}</div>"
                f"<p style='margin:4px 0 6px 0'><b>Vì sao được đưa vào danh sách:</b> điểm {result['score'] * 100:.1f}/100 "
                f"{'đạt/vượt' if flagged else 'dưới'} ngưỡng cảnh báo {result['threshold'] * 100:.1f}/100.</p>"
                f"<p style='margin:0 0 6px 0'><b>Gợi ý xử lý:</b> {html.escape(action)}</p>"
                f"<p style='margin:0'><b>Thông tin đối chiếu:</b></p>"
                f"<ul style='margin:2px 0 6px 0'>{items}</ul>"
                f"<p style='margin:0; font-size:12px; color:{th.MUTED}'>"
                "Điểm không phải xác suất đã hiệu chỉnh; thông tin đối chiếu không xác nhận nguyên nhân.</p>")

    def _page_overview(self):
        page, lay = self._page()
        p, m = self.predictor, self.metrics
        row = QHBoxLayout()
        row.setSpacing(16)
        self.kpiOrders = th.Kpi("Đơn hàng trong hệ thống", f"{len(p.order_ids()):,}",
                                "đơn có đặc trưng để dự đoán")
        self.kpiLate = th.Kpi("Đơn cần kiểm tra giao hàng", f"{int(self.pred['delivery']['prediction'].sum()):,}",
                              "cảnh báo trên tập kiểm thử lịch sử", "orange")
        self.kpiLow = th.Kpi("Đơn cần chăm sóc sau giao", f"{int(self.pred['review']['prediction'].sum()):,}",
                             "cảnh báo trên tập kiểm thử lịch sử", "orange")
        self.kpiSellers = th.Kpi("Nhà bán có dữ liệu lịch sử", "–", "")
        for k in (self.kpiOrders, self.kpiLate, self.kpiLow, self.kpiSellers):
            row.addWidget(k)
        lay.addLayout(row)

        mid = QHBoxLayout()
        mid.setSpacing(16)
        box, bl = th.card("Tra cứu nhanh đơn hàng")
        edit, predict = self._order_input()
        r = QHBoxLayout()
        r.addWidget(edit, 1)
        r.addWidget(predict)
        bl.addLayout(r)
        bl.addWidget(self._order_count())
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.quick = {}
        for key, title in (("delivery", "Theo dõi giao hàng"), ("review", "Chăm sóc sau giao"),
                           ("seller", "Đặc điểm nhà bán")):
            tile = th.Kpi(title, "–", "chưa tra cứu", "orange" if key != "seller" else "navy")
            tile.note.setWordWrap(True)
            tile.note.setMinimumHeight(36)
            tile.note.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.quick[key] = tile
            tiles.addWidget(tile)
        bl.addLayout(tiles)
        mid.addWidget(box)
        lay.addLayout(mid)

        tabs = QTabWidget()
        actions = QWidget()
        al = QHBoxLayout(actions)
        for key, title in (("delivery", "Kiểm tra giao hàng"), ("review", "Chăm sóc sau giao")):
            box, bl = th.card(title, "Dữ liệu kiểm thử lịch sử; nhấp đúp một đơn để xem chi tiết.")
            bl.addWidget(self._priority_table(key))
            al.addWidget(box)
        tabs.addTab(actions, "Danh sách ưu tiên")
        box, bl = th.card("Tình hình chung trên tập kiểm thử lịch sử")
        self.figOverview, canvas = make_canvas(240)
        bl.addWidget(canvas, 1)
        tabs.addTab(box, "Phân tích dữ liệu")
        lay.addWidget(tabs, 1)
        btn = QPushButton("So sánh và lựa chọn nhà bán →")
        btn.setObjectName("primary")
        btn.clicked.connect(lambda: self.goto(3))
        lay.addWidget(btn)
        return page

    def refreshOverview(self):
        p, c = self.predictor, self.cluster
        q = c["quality"]
        self.kpiSellers.set(f"{q['Số nhà bán']:,}", "từ 5 đơn hoàn tất; xem tiêu chí đề xuất ở Hiệu quả nhà bán")
        fig = self.figOverview
        fig.clear()
        axes = fig.subplots(1, 3)
        for ax, key, title in ((axes[0], "delivery", "Điểm rủi ro giao trễ"),
                               (axes[1], "review", "Điểm rủi ro đánh giá 1–2 sao")):
            ax.hist(self.pred[key]["score"], bins=40, color=th.NAVY, alpha=0.9)
            ax.axvline(self.threshold[key], color=th.ORANGE, linewidth=2,
                       label=f"Ngưỡng {self.threshold[key]:.3f}")
            ax.set_title(title)
            ax.set_xlabel("Điểm rủi ro")
            ax.set_ylabel("Số đơn")
            ax.legend(fontsize=8, frameon=False)
            style_axis(ax)
        sizes = c["means"]["Số nhà bán"]
        axes[2].bar(sizes.index, sizes.values, color=th.CLUSTER_COLORS[:len(sizes)], width=0.6)
        for i, v in enumerate(sizes.values):
            axes[2].text(i, v, f"{v:,} ({v / sizes.sum():.1%})", ha="center", va="bottom", fontsize=9,
                         color=th.TEXT)
        axes[2].set_title("Số nhà bán hàng theo nhóm")
        axes[2].set_ylabel("Số nhà bán hàng")
        axes[2].margins(y=0.15)
        style_axis(axes[2])
        axes[2].grid(axis="x", visible=False)
        fig.canvas.draw_idle()

    # ================================================================ 6.1.2 / 6.1.3
    def _page_model(self, key):
        task, ui, m = TASKS[key], self.ui[key], self.metrics[key]
        page, lay = self._page()
        body = QHBoxLayout()
        body.setSpacing(16)
        lay.addLayout(body)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(16)
        box, bl = th.card("Tra cứu đơn cần kiểm tra")
        edit, predict = self._order_input()
        r = QHBoxLayout()
        r.addWidget(edit, 1)
        r.addWidget(predict)
        bl.addLayout(r)
        bl.addWidget(self._order_count())
        left.addWidget(box)

        box, bl = th.card("Cảnh báo hỗ trợ xử lý", task["when"])
        ui["result"] = th.RiskResult()
        bl.addWidget(ui["result"])
        left.addWidget(box)

        box, bl = th.card("Thông tin đơn hàng")
        ui["info"] = th.make_table()
        bl.addWidget(ui["info"])
        left.addWidget(box, 1)
        holder = QWidget()
        holder.setLayout(left)
        holder.setFixedWidth(420)
        body.addWidget(holder)

        # --- 6.2.2 / 6.2.3 kết quả mô hình
        box, outer = th.card(f"Kết quả mô hình trên tập kiểm thử – {self.chosen[key]}")
        ui["stack"] = QStackedWidget()
        ui["stack"].addWidget(self._empty("Chưa chọn đơn hàng.\nChọn một đơn ở ô bên trái để xem kết quả dự đoán "
                                          "và kết quả của mô hình."))
        content = QWidget()
        bl = QVBoxLayout(content)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        hint = QLabel(f"{m['n']:,} đơn từ 01/05/2018; tỷ lệ {task['event']} thực tế {m['base']:.2%}, "
                      f"mô hình cảnh báo {m['flagged']:.1%} số đơn.")
        hint.setObjectName("cardHint")
        bl.addWidget(hint)
        ui["stack"].addWidget(content)
        outer.addWidget(ui["stack"], 1)
        strip = QHBoxLayout()
        strip.setSpacing(10)
        for label, value, accent in (("Recall", f"{m['recall']:.3f}", "orange"),
                                     ("Precision", f"{m['precision']:.3f}", "navy"),
                                     ("F1", f"{m['f1']:.3f}", "navy"),
                                     ("Average Precision", f"{m['ap']:.3f}", "navy"),
                                     ("ROC-AUC", f"{m['roc']:.3f}", "navy")):
            strip.addWidget(th.Kpi(label, value, accent=accent, small=True))
        bl.addLayout(strip)
        tabs = QTabWidget()
        ui["figCm"], canvas = make_canvas()
        tabs.addTab(canvas, "Ma trận nhầm lẫn")
        ui["figPr"], canvas = make_canvas()
        tabs.addTab(canvas, "Đường Precision–Recall")
        ui["compare"] = th.make_table()
        tabs.addTab(ui["compare"], "So sánh 4 mô hình")
        ui["tabs"] = tabs
        bl.addWidget(tabs, 1)
        workflow = QTabWidget()
        operation = QWidget()
        ol = QVBoxLayout(operation)
        ui["guidance"] = QLabel("Chọn một đơn để xem thông tin đối chiếu và gợi ý xử lý.")
        ui["guidance"].setWordWrap(True)
        ui["guidance"].setTextFormat(Qt.TextFormat.RichText)
        ol.addWidget(ui["guidance"])
        hint = QLabel("Danh sách ưu tiên · nhấp đúp một dòng để mở đơn")
        hint.setObjectName("muted")
        ol.addWidget(hint)
        ol.addWidget(self._priority_table(key), 1)
        workflow.addTab(operation, "Kiểm tra và xử lý")
        workflow.addTab(box, "Đánh giá mô hình")
        body.addWidget(workflow, 1)
        return page

    def showModelResults(self, key):
        task, ui, m, df = TASKS[key], self.ui[key], self.metrics[key], self.pred[key]
        fig = ui["figCm"]
        fig.clear()
        ax = fig.add_subplot(111)
        cm = m["cm"]
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=NAVY_CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%} của hàng", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white" if share[i, j] > 0.5 else th.NAVY)
        ax.set_xticks([0, 1], task["labels"])
        ax.set_yticks([0, 1], task["labels"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(f"Bỏ sót {cm[1, 0]:,} đơn {task['event']}, báo nhầm {cm[0, 1]:,} đơn")
        ax.tick_params(colors=th.TEXT, labelsize=10, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        style_title(ax)
        fig.canvas.draw_idle()

        fig = ui["figPr"]
        fig.clear()
        ax = fig.add_subplot(111)
        prec, rec, _ = precision_recall_curve(df["target"], df["score"])
        ax.plot(rec, prec, color=th.NAVY, linewidth=2, label=f"{self.chosen[key]} (AP = {m['ap']:.3f})")
        ax.axhline(m["base"], color=th.MUTED, linestyle="--", linewidth=1.2,
                   label=f"Đoán ngẫu nhiên = {m['base']:.3f}")
        ax.scatter([m["recall"]], [m["precision"]], s=90, color=th.ORANGE, zorder=5, edgecolor="white",
                   linewidth=1.5, label=f"Ngưỡng đang dùng {self.threshold[key]:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1)
        ax.set_title(f"Đường Precision–Recall trên tập kiểm thử – {task['event']}")
        ax.legend(frameon=False, fontsize=9)
        style_axis(ax)
        fig.canvas.draw_idle()

        comp = pd.read_csv(RESULT_DIR / f"{FILES[key]}_comparison.csv")
        if key == "delivery":
            comp["model"] = comp["model"].map(MODEL_VI)
            comp = comp.rename(columns={"model": "Mô hình", "split": "Tập", "AP": "Average Precision",
                                        "ROC_AUC": "ROC-AUC", "threshold": "Ngưỡng"})
        else:
            comp = comp.rename(columns={"Precision lớp 1": "Precision", "Recall lớp 1": "Recall",
                                        "F1 lớp 1": "F1"})
        comp = comp[["Mô hình", "Tập", "Average Precision", "ROC-AUC", "Precision", "Recall", "F1",
                     "Accuracy", "Ngưỡng", "TN", "FP", "FN", "TP"]]
        comp = comp.sort_values(["Tập", "Average Precision"], ascending=[False, False]).reset_index(drop=True)
        chosen = [i for i, name in enumerate(comp["Mô hình"]) if name == self.chosen[key]]
        th.fill_table(ui["compare"], comp, highlight_rows=chosen, stretch_last=False)

    # ================================================================ tra cứu đơn
    def lookup(self, order_id):
        order_id = str(order_id).split("·")[0].strip()        # mục trong danh sách có dạng "mã đơn · cụm"
        if not order_id:
            QMessageBox.warning(self, "Chưa chọn đơn hàng", "Chọn một đơn trong danh sách hoặc gõ mã đơn hàng.")
            return
        try:
            order_id = self.predictor.order_codes.resolve(order_id)
            info = self.predictor.order_info(order_id)
        except KeyError:
            QMessageBox.warning(self, "Không tìm thấy đơn hàng",
                                f"Không có đơn {order_id} trong dữ liệu. Kiểm tra lại mã đơn, "
                                "hoặc chọn một đơn có sẵn trong danh sách.")
            return
        self.order_id = order_id
        for key in TASKS:
            self.ui[key]["stack"].setCurrentIndex(1)
        for edit in self.orderEdits:
            edit.setEditText(self.predictor.order_codes.display(order_id))
            edit.setToolTip(f"Mã gốc: {order_id}")
            edit.lineEdit().setCursorPosition(0)      # hiện phần đầu mã đơn thay vì phần cuối
        results = {"delivery": self.predictor.predict_delivery(order_id),
                   "review": self.predictor.predict_review(order_id)}
        for key, res in results.items():
            task, ui = TASKS[key], self.ui[key]
            ui["result"].setResult(res, task["positive"], task["negative"], task["actual"])
            ui["guidance"].setText(self._guidance(key, res, order_id))
            th.fill_key_value(ui["info"], info, raw_ids={"Mã đơn hàng": order_id})
            tile = self.quick[key]
            if res is None:
                tile.set("–", "đơn không có đặc trưng cho mô hình này")
            else:
                verdict = task["positive"] if res["label"] == 1 else task["negative"]
                tile.set(verdict, f"Điểm cảnh báo {res['score'] * 100:.1f}/100 · xem hướng xử lý ở màn hình chi tiết")
        sellers = self.sellersOfOrder(order_id)
        th.fill_table(self.tableOrderSellers, sellers, decimals=2)
        if len(sellers) == 1:
            s = sellers.iloc[0]
            self.quick["seller"].set("1 nhà bán", str(s["Tên nhóm"]))
        else:
            groups = "; ".join(sellers["Tên nhóm"].drop_duplicates())
            self.quick["seller"].set(f"{len(sellers)} nhà bán", groups)
        self.labelOrderSellers.setText(f"Đơn {self.predictor.order_codes.display(order_id)} có {len(sellers)} nhà bán hàng.")

    def sellersOfOrder(self, order_id):
        """Nhà bán của đơn, gán theo kết quả K-Means đang hiển thị (K hiện tại)."""
        sellers = self.predictor.sellers_of_order(order_id).rename(columns={"Tên cụm": "Tên nhóm"})
        current = self.cluster["clusters"].set_index("seller_id")
        for i, sid in enumerate(sellers["seller_id"]):
            if sid in current.index:
                sellers.loc[i, "Cụm"] = current.loc[sid, "cluster"]
                sellers.loc[i, "Tên nhóm"] = current.loc[sid, "cluster_name"]
        sellers["Mã nhà bán"] = sellers["seller_id"].map(self.predictor.seller_codes.display)
        show = sellers[["Mã nhà bán", "Bang", "Cụm", "Tên nhóm", "Tiền hàng", "Lịch sử seller"]].rename(
            columns={"Lịch sử seller": "Lịch sử nhà bán", "Bang": "Bang nhà bán (Brazil)"})
        show.attrs["raw_ids"] = {"Mã nhà bán": sellers["seller_id"].tolist()}
        return show

    # ================================================================ 6.1.4 Phân nhóm
    def _page_sellers(self):
        page, lay = self._page()
        tabs = QTabWidget()
        main = QWidget()
        ml = QVBoxLayout(main)
        box, bl = th.card("Đề xuất theo tiêu chí bạn chọn", "Dựa trên lịch sử toàn kỳ.")
        row = QHBoxLayout()
        self.sellerCriterion = QComboBox()
        self.sellerCriterion.addItems(CRITERIA)
        self.sellerState = QComboBox()
        self.sellerState.addItem("Tất cả bang", "")
        for code in sorted(self.predictor.sellers["seller_state"].unique()):
            self.sellerState.addItem(state_label(code), code)
        self.sellerSearch = QLineEdit()
        self.sellerSearch.setPlaceholderText("Tìm mã nhà bán SEL-… hoặc mã gốc")
        row.addWidget(QLabel("Ưu tiên"))
        row.addWidget(self.sellerCriterion)
        row.addWidget(self.sellerState)
        row.addWidget(self.sellerSearch, 1)
        bl.addLayout(row)
        self.sellerRecommendation = QLabel()
        self.sellerRecommendation.setWordWrap(True)
        self.sellerRecommendation.setTextFormat(Qt.TextFormat.PlainText)
        bl.addWidget(self.sellerRecommendation)
        hint = QLabel("Chỉ đề xuất nhà bán có từ 20 đơn (từ 10 đánh giá với tiêu chí đánh giá tốt).")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        bl.addWidget(hint)
        ml.addWidget(box)
        self.sellerRanking = th.make_table()
        self.sellerRanking.setSelectionMode(self.sellerRanking.SelectionMode.ExtendedSelection)
        ml.addWidget(self.sellerRanking, 2)
        row = QHBoxLayout()
        row.addWidget(QLabel("Giữ Ctrl và chọn 2–3 nhà bán trong bảng để đối chiếu."))
        compare = QPushButton("So sánh nhà bán đã chọn")
        compare.setObjectName("primary")
        compare.clicked.connect(self.compareSelectedSellers)
        row.addWidget(compare)
        ml.addLayout(row)
        self.sellerComparison = th.make_table()
        self.sellerComparison.setMaximumHeight(185)
        ml.addWidget(self.sellerComparison, 1)
        self.comparisonHint = QLabel("Chưa chọn nhà bán để so sánh.")
        self.comparisonHint.setWordWrap(True)
        self.comparisonHint.setTextFormat(Qt.TextFormat.PlainText)
        ml.addWidget(self.comparisonHint)
        tabs.addTab(main, "Lựa chọn và so sánh")
        tabs.addTab(self._page_sellers_advanced(), "Phân tích nhóm chuyên sâu")
        lay.addWidget(tabs)
        for widget in (self.sellerCriterion, self.sellerState):
            widget.currentIndexChanged.connect(self.refreshSellerRanking)
        self.sellerSearch.textChanged.connect(self.refreshSellerRanking)
        return page

    def _seller_display(self, df):
        return pd.DataFrame({
            "Mã nhà bán": df["seller_id"].map(self.predictor.seller_codes.display),
            "Bang": df["seller_state"].map(state_label),
            "Số đơn": df["completed_order_count"].astype(int),
            "Số đánh giá": df["reviewed_order_count"].astype(int),
            "Đúng hẹn": df["late_order_rate"].map(lambda v: f"{1-v:.1%}" if pd.notna(v) else "Chưa có"),
            "Giao TB (ngày)": df["mean_delivery_days"].round(1),
            "Đánh giá 1–2 sao": df["low_review_rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "Chưa có"),
            "Điểm sao TB": df["mean_review_score"].round(2),
            "Cơ sở đề xuất": df["eligible"].map({True: "Đủ dữ liệu theo quy tắc", False: "Ít / thiếu dữ liệu"}),
        })

    def refreshSellerRanking(self):
        if not hasattr(self, "sellerHistory"):
            return
        # Bổ sung cả nhà bán dưới 5 đơn; không suy diễn dữ liệu không có.
        history = self.predictor.sellers.reset_index()[["seller_id", "seller_state"]].merge(
            self.sellerHistory.drop(columns="seller_state"), on="seller_id", how="left")
        for col in ("completed_order_count", "reviewed_order_count"):
            history[col] = history[col].fillna(0)
        df = rank_sellers(history, self.sellerCriterion.currentText())
        state = self.sellerState.currentData()
        if state:
            df = df[df["seller_state"].eq(state)]
        text = self.sellerSearch.text().strip().lower()
        if text:
            codes = df["seller_id"].map(self.predictor.seller_codes.display).str.lower()
            df = df[df["seller_id"].str.contains(text, regex=False) | codes.str.contains(text, regex=False)]
        self.rankedSellers = df.reset_index(drop=True)
        show = self._seller_display(self.rankedSellers)
        show.attrs["raw_ids"] = {"Mã nhà bán": self.rankedSellers["seller_id"].tolist()}
        th.fill_table(self.sellerRanking, show, decimals=1)
        th.fill_table(self.sellerComparison, pd.DataFrame())
        self.comparisonHint.setText("Chọn 2–3 nhà bán để so sánh. Thứ tự bảng theo tiêu chí đang chọn.")
        candidates = self.rankedSellers[self.rankedSellers["eligible"]]
        if candidates.empty:
            self.sellerRecommendation.setText("Chưa có nhà bán đủ dữ liệu để đề xuất.")
        else:
            top = candidates.iloc[0]
            code = self.predictor.seller_codes.display(top["seller_id"])
            metric = CRITERIA[self.sellerCriterion.currentText()][0]
            value = f"{top['adjusted']:.1f} ngày" if metric == "mean_delivery_days" else f"{top['adjusted']:.1%}"
            label = {"late_order_rate": "tỷ lệ trễ", "mean_delivery_days": "thời gian giao", "low_review_rate": "tỷ lệ 1–2 sao"}[metric]
            self.sellerRecommendation.setText(
                f"Đề xuất: {code} · {int(top['completed_order_count'])} đơn, "
                f"{int(top['reviewed_order_count'])} đánh giá · {label} {value} (sau điều chỉnh).")

    def compareSelectedSellers(self):
        rows = sorted(i.row() for i in self.sellerRanking.selectionModel().selectedRows())
        if not 2 <= len(rows) <= 3:
            self.comparisonHint.setText("Hãy giữ Ctrl và chọn từ 2 đến 3 nhà bán để so sánh.")
            return
        selected = self.rankedSellers.iloc[rows]
        th.fill_table(self.sellerComparison, self._seller_display(selected), decimals=1)
        for c in range(self.sellerComparison.columnCount() - 1):      # cột của hai bảng thẳng hàng nhau
            self.sellerComparison.setColumnWidth(c, self.sellerRanking.columnWidth(c))
        eligible = selected[selected["eligible"]]
        if eligible.empty:
            self.comparisonHint.setText("Các nhà bán đã chọn chưa đủ dữ liệu theo quy tắc; chưa đưa ra đề xuất.")
        else:
            code = self.predictor.seller_codes.display(eligible.iloc[0]["seller_id"])
            self.comparisonHint.setText(f"Trong các nhà bán đã chọn, ưu tiên {code} theo tiêu chí "
                                        f"{self.sellerCriterion.currentText().lower()} sau điều chỉnh số mẫu. "
                                        "Đối chiếu thêm các chỉ số còn lại trong bảng trước khi lựa chọn.")

    def _page_sellers_advanced(self):
        page, lay = self._page()
        body = QHBoxLayout()
        body.setSpacing(16)
        lay.addLayout(body)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(16)
        box, bl = th.card("Thiết lập K-Means")
        r = QHBoxLayout()
        self.comboBoxK = QComboBox()
        for k in sc.K_VALUES:
            self.comboBoxK.addItem(f"K = {k}", k)
        self.comboBoxK.setCurrentIndex(self.comboBoxK.findData(self.predictor.kmeans["k"]))
        self.pushButtonRunKMeans = QPushButton("Chạy phân nhóm")
        self.pushButtonRunKMeans.setObjectName("primary")
        self.pushButtonRunKMeans.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pushButtonRunKMeans.clicked.connect(self.runKMeans)
        r.addWidget(QLabel("Số cụm K"))
        r.addWidget(self.comboBoxK)
        r.addWidget(self.pushButtonRunKMeans, 1)
        bl.addLayout(r)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.kpiSil = th.Kpi("Silhouette", small=True, accent="orange")
        self.kpiDbi = th.Kpi("Davies–Bouldin", small=True)
        self.kpiCh = th.Kpi("Calinski–Harabasz", small=True)
        self.kpiAri = th.Kpi("ARI lấy mẫu lại", small=True)
        for i, k in enumerate((self.kpiSil, self.kpiDbi, self.kpiCh, self.kpiAri)):
            grid.addWidget(k, i // 2, i % 2)
        bl.addLayout(grid)
        self.labelClusterInfo = QLabel("Chọn số cụm K rồi bấm Chạy phân nhóm.")
        self.labelClusterInfo.setObjectName("muted")
        self.labelClusterInfo.setWordWrap(True)
        bl.addWidget(self.labelClusterInfo)
        left.addWidget(box)

        box, bl = th.card("Nhà bán hàng của đơn đang tra cứu")
        r = QHBoxLayout()
        edit, predict = self._order_input()
        predict.setText("Tra cứu")
        r.addWidget(edit, 1)
        r.addWidget(predict)
        bl.addLayout(r)
        bl.addWidget(self._order_count())
        self.labelOrderSellers = QLabel("Chưa tra cứu đơn nào. Chọn một đơn để xem nhóm của từng nhà bán hàng.")
        self.labelOrderSellers.setObjectName("muted")
        self.labelOrderSellers.setWordWrap(True)
        bl.addWidget(self.labelOrderSellers)
        self.tableOrderSellers = th.make_table()
        bl.addWidget(self.tableOrderSellers, 1)
        left.addWidget(box, 1)
        holder = QWidget()
        holder.setLayout(left)
        holder.setFixedWidth(420)
        body.addWidget(holder)

        # --- 6.2.4 kết quả phân nhóm
        box, outer = th.card("Kết quả phân nhóm nhà bán hàng")
        self.stackSeller = QStackedWidget()
        self.stackSeller.addWidget(self._empty("Chưa chạy phân nhóm.\nChọn số cụm K rồi bấm Chạy phân nhóm để xem "
                                               "kết quả."))
        content = QWidget()
        bl = QVBoxLayout(content)
        bl.setContentsMargins(0, 0, 0, 0)
        self.stackSeller.addWidget(content)
        outer.addWidget(self.stackSeller, 1)
        tabs = QTabWidget()
        self.figProfile, canvas = make_canvas(230)
        self.tableProfile = th.make_table()
        self.tableProfile.setMaximumHeight(150)
        tabs.addTab(self._stacked(canvas, self.tableProfile, 3, 1), "Hồ sơ nhóm")

        self.figElbow, canvas = make_canvas(340)
        self.tableEval = th.make_table()
        self.tableEval.setMaximumHeight(190)
        tabs.addTab(self._stacked(canvas, self.tableEval, 3, 2), "Chọn K")

        self.figQuality, canvas = make_canvas(300)
        self.tableQuality = th.make_table()
        self.tableQuality.setMaximumHeight(150)
        tabs.addTab(self._stacked(canvas, self.tableQuality, 3, 1), "Chất lượng, ổn định")

        self.figScatter, canvas = make_canvas(320)
        w = QWidget()
        wl = QVBoxLayout(w)
        wl.setContentsMargins(0, 10, 0, 0)
        r = QHBoxLayout()
        self.comboBoxView = QComboBox()
        self.comboBoxView.addItems(["Minh họa bằng PCA (2 chiều)", "Phân bố đặc trưng gốc theo nhóm"])
        self.comboBoxView.currentIndexChanged.connect(self.showScatter)
        r.addWidget(QLabel("Kiểu biểu đồ"))
        r.addWidget(self.comboBoxView, 1)
        wl.addLayout(r)
        wl.addWidget(canvas, 1)
        tabs.addTab(w, "Phân bố")

        w = QWidget()
        wl = QVBoxLayout(w)
        wl.setContentsMargins(0, 10, 0, 0)
        r = QHBoxLayout()
        self.lineEditSellerFilter = QLineEdit()
        self.lineEditSellerFilter.setPlaceholderText("Mã SEL-0001 / mã gốc / bang (SP, Sao Paulo)")
        self.comboBoxClusterFilter = QComboBox()
        self.labelSellerCount = QLabel()
        self.labelSellerCount.setObjectName("muted")
        r.addWidget(self.lineEditSellerFilter, 1)
        r.addWidget(self.comboBoxClusterFilter)
        r.addWidget(self.labelSellerCount)
        wl.addLayout(r)
        self.tableSellers = th.make_table()
        wl.addWidget(self.tableSellers, 1)
        self.lineEditSellerFilter.textChanged.connect(self.showSellerList)
        self.comboBoxClusterFilter.currentIndexChanged.connect(self.showSellerList)
        tabs.addTab(w, "Danh sách nhà bán")
        self.tabsSeller = tabs
        bl.addWidget(tabs, 1)
        body.addWidget(box, 1)
        return page

    def _stacked(self, canvas, table, a=1, b=1):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 10, 0, 0)
        lay.addWidget(canvas, a)
        lay.addWidget(table, b)
        return w

    def runKMeans(self):
        try:
            self.cluster = sc.fit(self.comboBoxK.currentData())
        except Exception as error:
            QMessageBox.critical(self, "Không chạy được K-Means",
                                 f"{error}\nKiểm tra các tệp CSV trong thư mục data/raw rồi chạy lại.")
            return
        c, q, saved = self.cluster, self.cluster["quality"], self.predictor.kmeans
        self.kpiSil.set(f"{q['Silhouette']:.4f}")
        self.kpiDbi.set(f"{q['Davies-Bouldin']:.4f}")
        self.kpiCh.set(f"{q['Calinski-Harabasz']:.1f}")
        self.kpiAri.set(f"{saved['ari_mean']:.4f}" if c["k"] == saved["k"] else "–")
        self.labelClusterInfo.setText(
            f"{q['Số nhà bán']:,} nhà bán có từ {sc.MIN_ORDERS} đơn hoàn tất đủ điều kiện, tổng hợp trên "
            f"{saved['eligible_orders']:,} đơn. {q['Số nhà bán Silhouette âm']} nhà bán "
            f"({q['Tỷ lệ Silhouette âm']:.1%}) có Silhouette âm."
            + ("" if c["k"] == saved["k"] else f" ARI lấy mẫu lại chỉ được tính cho K = {saved['k']} của báo cáo."))
        self.comboBoxClusterFilter.blockSignals(True)
        self.comboBoxClusterFilter.clear()
        self.comboBoxClusterFilter.addItems(["Tất cả nhóm"] + c["codes"])
        self.comboBoxClusterFilter.blockSignals(False)
        self.showProfile()
        self.showElbow()
        self.showQuality()
        self.showScatter()
        self.showSellerList()
        self.refreshOverview()
        self.stackSeller.setCurrentIndex(1)
        if self.order_id:
            th.fill_table(self.tableOrderSellers, self.sellersOfOrder(self.order_id), decimals=2)

    def showProfile(self):
        c = self.cluster
        means = c["means"].reset_index().rename(columns={**FEATURE_VI, "mean_review_score": "Điểm đánh giá TB"})
        th.fill_table(self.tableProfile, means, decimals=3)
        fig = self.figProfile
        fig.clear()
        ax = fig.add_subplot(111)
        z = c["centroids"]
        limit = max(float(np.abs(z.to_numpy()).max()), 0.01)
        image = ax.imshow(z, cmap=DIVERGING, aspect="auto", vmin=-limit, vmax=limit)
        for i in range(len(z)):
            for j in range(z.shape[1]):
                v = z.iloc[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if abs(v) > limit * 0.55 else th.TEXT)
        labels = ["Số đơn\n(log)", "Giá trị/đơn\n(log)", "Tỷ lệ\ngiao trễ", "Thời gian\ngiao TB", "Tỷ lệ đánh\ngiá thấp"]
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(z)), z.index)
        ax.tick_params(colors=th.TEXT, labelsize=10, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Hồ sơ tâm cụm trên năm đặc trưng đã biến đổi và chuẩn hóa")
        style_title(ax)
        bar = fig.colorbar(image, ax=ax, pad=0.02)
        bar.set_label("Độ lệch chuẩn so với trung bình", fontsize=9, color=th.MUTED)
        bar.ax.tick_params(labelsize=8, colors=th.MUTED)
        bar.outline.set_visible(False)
        fig.canvas.draw_idle()

    def showElbow(self):
        c, rep = self.cluster, self.k_report
        th.fill_table(self.tableEval, rep, highlight_rows=[int(c["k"] - rep["K"].min())], stretch_last=False)
        fig = self.figElbow
        fig.clear()
        axes = fig.subplots(2, 2, sharex=True).ravel()
        for ax, col, title in zip(axes, ["Inertia", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
                                  ["Elbow (Inertia)", "Silhouette trung bình", "Davies–Bouldin (thấp tốt hơn)",
                                   "Calinski–Harabasz (cao tốt hơn)"]):
            ax.plot(rep["K"], rep[col], marker="o", markersize=4, color=th.NAVY, linewidth=2)
            ax.axvline(c["k"], color=th.ORANGE, linestyle="--", linewidth=1.8, label=f"K = {c['k']}")
            ax.set_title(title)
            ax.legend(frameon=False, fontsize=8)
            style_axis(ax)
            ax.title.set_fontsize(10)
        for ax in axes[2:]:
            ax.set_xlabel("Số cụm K")
        fig.canvas.draw_idle()

    def showQuality(self):
        c = self.cluster
        th.fill_table(self.tableQuality, c["cluster_quality"])
        fig = self.figQuality
        fig.clear()
        same_k = c["k"] == self.predictor.kmeans["k"]
        axes = fig.subplots(1, 2) if same_k else [fig.add_subplot(111)]
        ax = axes[0]
        y_lower = 10
        for i, code in enumerate(c["codes"]):
            values = np.sort(c["clusters"].loc[c["clusters"]["cluster"] == code, "silhouette"].to_numpy())
            ax.fill_betweenx(np.arange(y_lower, y_lower + len(values)), 0, values,
                             color=th.CLUSTER_COLORS[i % 10], alpha=0.9)
            ax.text(-0.38, y_lower + len(values) / 2, f"{code}\nn = {len(values):,}", fontsize=9,
                    va="center", color=th.TEXT)
            y_lower += len(values) + 10
        mean = c["quality"]["Silhouette"]
        ax.axvline(mean, color=th.MUTED, linestyle="--", linewidth=1.4)
        ax.set_xlim(-0.4, 0.8)
        ax.set_yticks([])
        ax.set_xlabel("Silhouette (nét đứt: trung bình)")
        ax.set_title(f"Silhouette từng nhà bán, trung bình {mean:.3f}")
        style_axis(ax)
        if same_k:
            ax, st = axes[1], self.stability
            ax.plot(st["Lần chạy"], st["ARI"], marker="o", markersize=4, color=th.NAVY, linewidth=1.8)
            ax.axhline(st["ARI"].mean(), color=th.ORANGE, linestyle="--", linewidth=1.8,
                       label=f"ARI trung bình = {st['ARI'].mean():.3f}")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Lần chạy")
            ax.set_ylabel("Adjusted Rand Index")
            ax.set_title("ARI qua 30 lần lấy mẫu 80%")
            ax.legend(frameon=False, fontsize=9, loc="lower left")
            style_axis(ax)
        fig.canvas.draw_idle()

    def showScatter(self):
        if self.cluster is None:
            return
        c = self.cluster
        fig = self.figScatter
        fig.clear()
        if self.comboBoxView.currentIndex() == 0:
            ax = fig.add_subplot(111)
            ids = c["clusters"]["cluster_id"].to_numpy()
            for i, code in enumerate(c["codes"]):
                mask = ids == i
                ax.scatter(c["pca"][mask, 0], c["pca"][mask, 1], s=18, alpha=0.65, color=th.CLUSTER_COLORS[i % 10],
                           edgecolor="white", linewidth=0.3, label=f"{code}: {c['names'][code]}")
            ax.set_xlabel(f"PC1 ({c['pca_var'][0]:.1%} phương sai)")
            ax.set_ylabel(f"PC2 ({c['pca_var'][1]:.1%} phương sai)")
            ax.set_title("Phân nhóm nhà bán hàng, minh họa bằng PCA")
            ax.legend(fontsize=8, frameon=False)
            style_axis(ax)
        else:
            axes = fig.subplots(1, 5)
            for ax, f in zip(axes, sc.FEATURE_COLS):
                values = [c["clusters"].loc[c["clusters"]["cluster"] == code, f].dropna().to_numpy()
                          for code in c["codes"]]
                box = ax.boxplot(values, showfliers=False, patch_artist=True, widths=0.6,
                                 medianprops={"color": "white", "linewidth": 2})
                for patch, i in zip(box["boxes"], range(len(values))):
                    patch.set_facecolor(th.CLUSTER_COLORS[i % 10])
                    patch.set_edgecolor(th.CLUSTER_COLORS[i % 10])
                ax.set_xticks(range(1, len(values) + 1), c["codes"])
                title = textwrap.fill(FEATURE_VI[f], 12)
                if f in sc.LOG_FEATURES:
                    ax.set_yscale("log")
                    title = FEATURE_VI[f].replace(" TB/đơn", "/đơn").replace(" hoàn tất", "") + "\n(trục log)"
                ax.set_title(title)
                style_axis(ax)
                ax.title.set_fontsize(9)
                ax.grid(axis="x", visible=False)
            fig.suptitle("Phân bố đặc trưng theo đơn vị gốc (ẩn điểm ngoại lệ)", fontsize=10, color=th.MUTED)
        fig.canvas.draw_idle()

    def showSellerList(self):
        if self.cluster is None:
            return
        df = self.cluster["clusters"].copy()
        df["display_id"] = df["seller_id"].map(self.predictor.seller_codes.display)
        df["state_label"] = df["seller_state"].map(state_label)
        text = search_text(self.lineEditSellerFilter.text().strip())
        if text:
            if text.upper() in STATE_NAMES:
                df = df[df["seller_state"].str.lower() == text]
            else:
                df = df[df["seller_id"].str.contains(text, regex=False)
                        | df["display_id"].str.lower().str.contains(text, regex=False)
                        | df["state_label"].map(search_text).str.contains(text, regex=False)]
        idx = self.comboBoxClusterFilter.currentIndex()
        if idx > 0:
            df = df[df["cluster"] == self.cluster["codes"][idx - 1]]
        self.labelSellerCount.setText(f"{len(df):,} nhà bán hàng")
        show = df[["display_id", "state_label", "cluster"] + sc.FEATURE_COLS + ["mean_review_score", "silhouette"]]
        show = show.rename(columns={**FEATURE_VI, "display_id": "Mã nhà bán", "state_label": "Bang nhà bán (Brazil)",
                                    "cluster": "Cụm", "mean_review_score": "Điểm đánh giá TB",
                                    "silhouette": "Silhouette"})
        show.attrs["raw_ids"] = {"Mã nhà bán": df["seller_id"].tolist()}
        th.fill_table(self.tableSellers, show, decimals=3, stretch_last=False)
