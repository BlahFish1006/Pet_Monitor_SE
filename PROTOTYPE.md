# Pet Edge Tracking System (PET) — 原型模擬說明書

> 本文件描述目前的**原型模擬(prototype simulation)**:以 **AI 生成影片**取代實體攝影機/麥克風/喇叭,在單機上跑完整的「感測 → 邊緣 AI 判定 → 警示/記錄」流程,並透過 n8n 呈現動作輸出階段。**尚未串接實體硬體**。

---

## 1. 系統概述

PET 是一套**本地、不依賴雲端**的邊緣運算寵物監控系統。核心理念:所有感測、AI 推論、異常判定、警示與記錄都在**本機(Local PC)**完成,零雲端、零頻繁誤報、即時本地嚇阻。

原型同時實作兩條獨立並行的偵測管線:

- **視覺管線**:攝影機畫面 → YOLO-World 物件偵測(狗/貓)→ 異常行為判定(進入禁區、進出畫面)。
- **聽覺管線**:麥克風/音訊 → 分貝分析 → 異常吠叫判定。

兩條管線各自持續運作、各自判定異常,判定出的事件以標準介面送往使用者介面(UI)與事件記錄。

---

## 2. 模擬範圍與限制(重要)

| 項目 | 原型現況 | 正式產品 |
|---|---|---|
| 影像來源 | **AI 生成影片檔**(`*.mp4`)經 `cv2.VideoCapture` 讀入 | 實體 HD Webcam 即時串流 |
| 音訊來源 | 影片內建音軌(PyAV 解碼)或本機麥克風 | 陣列麥克風(Array Mic) |
| 警報輸出 | 螢幕 banner + OS 通知 + n8n 工作流 | 實體本地喇叭/警示燈 |
| 禁區定義 | 影片中**畫好的紅框**,系統用顏色自動偵測;或 CLI 座標 | 使用者於 UI 圈選的虛擬電子圍籬 |
| 運算裝置 | CPU(`torch 2.x+cpu`),YOLO 約 0.7 s/幀 | 邊緣 GPU,即時 25 FPS |
| 電源管理 (B.PWR) | 未實作 | UPS / 市電-電池切換 |
| 儲存管理 (B.STORAGE) | 寫入 `events.log` 純文字 | 日誌 FIFO 循環覆蓋、容量監控 |

**結論**:原型驗證的是「**軟體與 AI 邏輯的端到端可行性**」,硬體整合、電源、儲存循環為後續工作。

---

## 3. 系統架構與 SE 對應

原型程式對應到系統工程文件中的功能方塊(Block)與介面(ICD):

| 方塊 | 名稱 | 對應原型實作 | 功能(Function) |
|---|---|---|---|
| **B.SENSE** | 影像與麥克風擷取 | `cv2.VideoCapture` + PyAV 音訊 / `sounddevice` | F.1.1 擷取影像、F.1.2 擷取音訊 |
| **B.COMP** | 邊緣運算處理單元(**Logic Core**) | `yolo_world_detector.py` + `bark_detector.py` + dashboard workers | F.2.1 影像物件辨識、F.2.2 音訊分貝辨識、F.2.3 影像後處理、F.2.4 分貝過載辨識 |
| **B.UI** | 警報器與使用者介面 | `pet_dashboard.py`(PySide6)+ n8n 工作流 | F.3 通知及使用介面、F.3.1 觸發本地警示通知 |
| **B.STORAGE** | 管理存儲空間資源 | `events.log` + n8n 日誌節點 | F.5 日誌(循環覆蓋為未來) |
| **B.PWR** | 管理用電資源 | 未實作 | F.4 電源管理 |

**介面(ICD):**

| 介面 ID | 訊號 | 內容 | 原型對應 |
|---|---|---|---|
| ICD-SENSE-COMP-001 | `RAW_AV_STREAM` | Video, Audio_dB | 影格 + 音訊 dBFS |
| **ICD-COMP-UI-001** | **`ALERT_TRIGGER`** | **Event_Type, Confidence_%, Timestamp** | n8n webhook 收到的 payload |
| ICD-COMP-STORAGE-001 | `STROE_STATE` | Source=STROE_SPACE | 事件寫入日誌 |
| ICD-PWR-COMP-001 | `PWR_STATE` | MAINS/BATT, BATT_Pct | 未實作 |

> **重要釐清**:異常的「判定」與「分類」都在 **B.COMP(Python)** 完成,事件送出時 `Event_Type` 已決定。n8n 端的 Switch 只是**依 `Event_Type` 派送(Event Dispatcher)**,不做判斷。

---

## 4. 偵測子系統(B.COMP / F.2)

### 4.1 視覺偵測 — `yolo_world_detector.py`

- **模型**:YOLO-World(`yolov8l-world`),開放詞彙(open-vocabulary)。以 `set_classes(["dog","cat"])` 客製成寵物偵測模型 `dogandcat.pt`。
- **詞彙限定**:只認 `dog` / `cat`。因此**人、家具等非寵物物件根本不會被偵測** → 自然排除誤報(對應 Scenario 0 的日常過濾)。
- **逐幀偵測**:每一(strided)幀執行 `model.predict()`,輸出每隻寵物的 bounding box + 類別 + 信心值,並即時畫框。
- **效能**:CPU 約 0.7 s/幀(imgsz=480)。以 `--stride N` 每 N 幀偵測一次、中間幀重用上次框,換取畫面流暢。

**異常行為判定(F.2.3 後處理):**

1. **禁區越界(Scenario 2)**:定義禁區 ROI;當任一寵物 bbox 與 ROI 相交即視為「進入」,觸發警報(碰到目標物前即預警)。禁區可由 `--danger-zone auto`(用 HSV 紅色遮罩自動偵測影片中畫好的紅框)或 `--danger-zone x1,y1,x2,y2`(正規化座標)指定。
2. **寵物進出畫面(Scenario 0)**:追蹤寵物在/不在畫面;離開 >2 秒 → `pet_out`,返回 → `pet_in`(含去抖動,避免漏偵測造成閃爍)。

### 4.2 聽覺偵測 — `bark_detector.py`

偵測「**相對於環境基準線的持續大聲**」,而非辨識狗叫音色:

| 參數 | 預設 | 意義 |
|---|---|---|
| `SAMPLE_RATE` / `FRAME_MS` | 16000 Hz / 100 ms | 音框取樣 |
| `BASELINE_WINDOW_S` | 30 s | 滾動基準線(取中位數,對突波穩健) |
| `LOUD_MARGIN_DB` | 8 dB | 高過基準線多少算「大聲」 |
| `DETECT_WINDOW_S` | 10 s | 持續性偵測窗 |
| `LOUD_RATIO` | 0.30 | 窗內需有多少比例的幀為大聲才觸發 |
| `COOLDOWN_S` | 30 s | 兩次警報最小間隔 |

流程:每 100 ms 算 RMS→dBFS → 滾動 30 s 中位數為 baseline → `frame_db > baseline + 8 dB` 即「大聲」→ 10 s 窗內 ≥30% 大聲 → 觸發 `abnormal_barking`。用「比例」而非「連續」是為了抓斷續吠叫(汪—汪—汪),同時忽略關門等一次性巨響。

---

## 5. 使用者介面 — `pet_dashboard.py`(B.UI / PySide6)

單一視窗整合雙核心,以 **QThread + Qt Signals** 讓視覺與聽覺執行緒互不阻塞:

```
AudioWorker (QThread) ──status / bark_alert──┐
                                             ├──▶ MainWindow (GUI thread)
VideoWorker (QThread) ──frame / pet_event────┘     即時影像 + 狀態面板 + 事件 + 警報
```

**畫面元素:**
- **即時影像**:annotated frame(含寵物框、禁區紅框)。
- **狀態面板**:Visual FPS、DOGS、CATS、AUDIO LEVEL、BASELINE、LOUD RATIO(長條)。
- **RECENT EVENTS**:事件縮圖列,依事件類型上色(吠叫/禁區=紅、外出=藍、返回=綠)。
- **警報 banner**:事件發生時跳出橫幅,並呼叫 OS 通知。

**音訊來源**:`--audio-source mic`(麥克風)或 `--audio-source video`(用影片自己的音軌,PyAV 解碼 → 16 kHz 單聲道 → 餵入同一套吠叫偵測)。

**事件日誌(B.STORAGE / F.5)**:所有事件附帶時間戳寫入 `events.log`:
```
2026-06-08 15:16:29 | danger_zone | 禁區警報 |
2026-06-08 15:16:40 | abnormal_barking | 異常吠叫 | Sustained loud audio ...
```

---

## 6. 動作輸出與 n8n(B.UI / Action Output)

### 6.1 介面 payload(ICD-COMP-UI-001 / ALERT_TRIGGER)

B.COMP 判定出事件後,以 HTTP POST 將 **ALERT_TRIGGER** 送到 n8n Webhook:

```json
{
  "interface": "ICD-COMP-UI-001",
  "signal": "ALERT_TRIGGER",
  "source_block": "B.COMP",
  "Event_Type": "danger_zone",
  "Confidence_%": 94,
  "Timestamp": "2026-06-08 15:16:29",
  "scenario": 2,
  "message": "Pet entered forbidden zone"
}
```
`Confidence_%` = 視覺事件取 YOLO 偵測框信心 ×100;吠叫取 `loud_ratio` ×100。

### 6.2 n8n 工作流 — `n8n_pet_workflow.json`

```
① B.SENSE → B.COMP(Python,示意節點)：Logic Core 在此做 F.2 判定
   F.1.1 影像 → F.2.1 → F.2.3 ┐
   F.1.2 音訊 → F.2.2 → F.2.4 ┴─▶ 產生 ICD-COMP-UI-001 ALERT_TRIGGER ⬇

② B.UI（n8n 實際執行）
   [Webhook: ICD-COMP-UI-001 ⇢ ALERT_TRIGGER]
        └▶ Event Dispatcher (by Event_Type)        ← 只派送，不判斷
             ├─ Scenario 1: 吠叫 → B.UI · F.3.1 異常吠叫警示 ┐
             ├─ Scenario 2: 越界 → B.UI · F.3.1 禁區越界警示 ┤→ B.STORAGE · F.5 日誌 (STROE_STATE)
             └─ fallback        → Unmatched
```

- 上游(攝影機→偵測→判定)用**示意節點 + 便利貼**呈現,說明這段在 Python 執行(n8n 跑不動即時 CV/音訊)。
- 下游(Webhook→Dispatcher→Action→Storage)為**真實運作**:demo 觸發時對應的線會即時亮起,並可在 Executions 分頁回放。

> **僅 `abnormal_barking` 與 `danger_zone` 送 n8n**;`pet_in`/`pet_out` 保留為本機事件(RECENT EVENTS + 日誌),不送 n8n,避免在路由 demo 中變成 Unmatched。

### 6.3 n8n 的定位:示意/可擴充的「動作輸出層」(誠實說明)

**在目前的原型中,動作輸出(通知 + 記錄)其實已由 Python 端完整完成**(banner、OS 通知、`events.log`),而 n8n 的 action 節點目前是 **placeholder(Set 節點)**,並未做 Python 還沒做的事 —— 因此**功能上與 Python 重疊**。

所以 n8n 在原型階段的定位是:

1. **視覺化呈現 Action Output 階段** —— demo 時讓人看到「事件流到系統的哪一個階段、走哪一條 scenario」。
2. **架構上把動作輸出做成可抽換、解耦的一層** —— 邊緣(B.COMP)只負責輸出標準的 `ALERT_TRIGGER`,「收到事件要做什麼」交給 n8n;改動作**不需更動邊緣程式**。

### 6.4 未來:把 placeholder 換成真實外部動作 → 成為真正的「外部整合層」

只要把每個 scenario 的 Set 節點換成 n8n 內建的整合節點,即可**不寫程式**讓事件扇出到外部服務,屆時 n8n 就**不再是 Python 的重複**,而是承擔 Python 不該自己硬寫的整合工作:

| 事件 | 未來 n8n 動作(取代 placeholder) | 可用 n8n 節點 |
|---|---|---|
| 異常吠叫 / 禁區越界 | 即時推播到飼主手機 | LINE / Telegram / Slack / Gmail |
| 任一事件 | 寫一列到雲端表單(時間、類型、Confidence_%、影像連結) | Google Sheets / Notion / Airtable |
| 禁區越界 | 觸發智慧家庭裝置(警示燈、智慧喇叭播安撫聲) | Home Assistant / IFTTT / HTTP |
| 嚴重或連續事件 | 發簡訊或自動撥號 | Twilio (SMS / Voice) |
| 所有事件 | 寫入資料庫供後續分析/報表 | Postgres / MySQL / HTTP Request |

**為什麼這樣 n8n 就有了「無法被 Python 順手取代」的理由:**

- **免寫程式的整合**:串 LINE / Email / Sheet 等第三方 API,用拉的就好,不必在邊緣裝置維護一堆 SDK、OAuth 金鑰與錯誤處理。
- **解耦與可維護**:通知對象、通道、訊息格式全在 n8n 調整,邊緣程式(B.COMP)完全不動。
- **非工程師可調整**:行為以 GUI 設定,不需改程式碼。
- **集中式 fan-out**:同一個事件可同時通知多方(手機推播 + 雲端表單 + 警示燈)。

**與 ICD 的關係**:無論未來動作怎麼擴充,邊緣端永遠只輸出 **ICD-COMP-UI-001 / `ALERT_TRIGGER`** 這個固定契約;新增的外部動作屬於 **B.UI 之外接整合**,介面不變。這正是把 payload 對齊成標準欄位(Event_Type / Confidence_% / Timestamp)的價值 —— **換動作不用改介面**。

---

## 7. 事件 ↔ 情境(Scenario)對應

| 事件 `Event_Type` | Scenario | 來源管線 | 動作 |
|---|---|---|---|
| `abnormal_barking` | 1 連續吠叫 | 聽覺 | 通知飼主 / 安撫聲 + 記錄 |
| `danger_zone` | 2 危險區 | 視覺 | 本地警告音 + 記錄 |
| `pet_in` / `pet_out` | 0 日常 | 視覺 | 本機記錄(進出) |
| (人/非寵物) | 0 日常 | 視覺 | 不偵測 → 零打擾(排除 FP) |
| — | 3 低資源 | — | **未實作** |

---

## 8. 測試素材(模擬影片)

| 檔案 | 內容 | 驗證的情境 |
|---|---|---|
| `test_mov.mp4` | 客廳,有人經過,金毛犬趴下後吠叫 | 客廳場景、人不誤報、吠叫 |
| `Locked_off_camera_fixed_tripo.mp4` | 玄關,大門開向院子,狗進出門口 | 攝影含門口、寵物進出 |
| `test_more.mp4` | 客廳,花瓶處畫紅色 Forbidden Zone,貓進入禁區,狗在旁 | 禁區自動偵測 + 越界警報 |
| `sample_av.mp4` | 合成投影片 + 安靜→大聲音軌 | 吠叫偵測觸發(可重現) |

> 影片為 AI 生成的監控視角素材;`make_test_clip.py` 可重新合成帶音軌的測試片。

---

## 9. 端到端資料流

```
[B.SENSE] 影片/麥克風
   │ ICD-SENSE-COMP-001 : RAW_AV_STREAM (影格 + 音訊dB)
   ▼
[B.COMP] 邊緣運算（Logic Core）
   ├─ 視覺：YOLO 偵測 → 禁區/進出判定
   └─ 聽覺：dBFS → baseline → 吠叫判定
   │ ICD-COMP-UI-001 : ALERT_TRIGGER (Event_Type, Confidence_%, Timestamp)
   ▼
[B.UI] 警示與介面
   ├─ Dashboard：banner / 狀態面板 / RECENT EVENTS
   └─ n8n：Event Dispatcher → F.3.1 動作
   │ ICD-COMP-STORAGE-001 : STROE_STATE
   ▼
[B.STORAGE] events.log / n8n 日誌節點
```

---

## 10. 已實作 vs 待辦

**已實作:**
- ✅ 視覺偵測(YOLO-World 狗/貓)+ 即時畫框與計數
- ✅ 聽覺吠叫偵測(滾動基準線 + 持續性窗)
- ✅ 禁區虛擬圍籬(自動偵測影片紅框 / 座標)
- ✅ 寵物進出畫面事件
- ✅ 非寵物自然排除(詞彙限定)→ 零誤報
- ✅ PySide6 統一儀表板(影像 + 狀態 + 事件 + 警報)
- ✅ 事件日誌 `events.log`
- ✅ n8n 動作輸出(ALERT_TRIGGER 對齊 ICD-COMP-UI-001,依 Event_Type 派送)

**待辦(後續):**
- ⬜ 串接實體硬體(Webcam / 陣列麥克風 / 喇叭)
- ⬜ B.PWR 電源管理(市電/電池切換,Scenario 3)
- ⬜ B.STORAGE 日誌 FIFO 循環覆蓋與容量監控
- ⬜ GPU 加速以達即時 25 FPS
- ⬜ 聲音分類器(如 YAMNet)區分狗叫與其他大聲響
- ⬜ n8n action 由 placeholder 升級為真實外部整合(LINE / Email / Google Sheet 等,見 §6.4),使其成為真正的「外部整合層」而非與 Python 重複

---

## 11. 執行方式

```powershell
conda activate pet_monitor
cd D:\_NYCU_course\System_Engineering\Pet_Monitor_SE

# 建立寵物偵測模型(首次,會下載 yolov8l-world.pt)
python yolo_world_detector.py --customize --save-model dogandcat.pt

# 統一儀表板:讀影片自身音軌 + 自動偵測禁區紅框 + 送 n8n
python pet_dashboard.py --source test_more.mp4 --model dogandcat.pt --stride 3 ^
    --audio-source video --danger-zone auto ^
    --n8n-webhook http://localhost:5678/webhook/pet-event
```

n8n 端:啟動 n8n → Import `n8n_pet_workflow.json` → 複製 Webhook Production URL → 將工作流切為 **Active**。

---

## 12. 程式檔案一覽

| 檔案 | 角色 |
|---|---|
| `yolo_world_detector.py` | 視覺核心(B.COMP)|
| `bark_detector.py` | 聽覺核心(B.COMP)|
| `pet_dashboard.py` | 統一儀表板 + 偵測整合(B.UI + B.COMP)|
| `n8n_client.py` | ALERT_TRIGGER 送出(ICD-COMP-UI-001)|
| `n8n_pet_workflow.json` | n8n 工作流(B.UI Action Output)|
| `make_test_clip.py` | 合成測試影片 |
| `requirements.txt` | 相依套件 |
| `events.log` | 事件日誌(B.STORAGE / 執行時產生)|
