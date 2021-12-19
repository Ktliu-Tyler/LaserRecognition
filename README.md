# **Laser_recognition模型製作與影像前置處理**
### **Author: Hdingzp4  Tylerj86**
* **影像資料抓取**：<br>
    * 使用colab:<br>
        我們的影像檔案都優先存儲於雲端硬碟中的1sec_video資料夾中，由於是使用colab進行編寫，我們引入google.colab.drive將colab掛載至雲端硬碟上以取得data並利於建立database。<br>
    * 使用jupyter:<br>
        只須直接使用本地影片路徑即可。

    * 影像處理:<br>
      * Video_process_tool:<br>
          為符合CNN LSTM模型所需訓練的格式，首先我們利用opencv-python模組進行影片的前置處理。
          我們建立了名為Video_process_tool的Class以利處理影像，於其中建立了get_mask resize_img及gray_img三種函式。
        * get_mask:<br>
            針對拍攝的影像設定一個固定的mask以清晰抓取的震動並切割。
        * resize_img:<br>
            將圖片縮放為63x75的影像。
        * gray_img:<br>
            利用opencv的cv2.cvtColor模組先將圖片轉成灰階。
      * 建立frames_extraction函式:<br>
            使用cv2抓取檔案中的影像，檢測影片抓取的幀數，加以切割並分配長度設定為15幀的影像。再將影像經前述的Video_proccess_tool處理後，對於每個pixel除以255，也就是使其成為介於0到1之間的數值，以方便後續進行卷積或是運算，最後將每幀圖片加入陣列中回傳。
      * 建立create_database函式:<br>
            走訪資料夾中的所有影像，調用frames_extraction並取得其回傳的影祥資料加入到同樣標籤的陣列中，分成相對應的features和Labels陣列分別代表該影片的data和其對應的標籤。

      * 資料拆分與下載:<br>
          使用sklearn對處理好的features和Labels，進行拆分，分成features_train(用於訓練的資料), features_test(用於測試的資料), labels_train(用於訓練的標籤), labels_test(用於測試的標籤), video_files_train(訓練的影片檔案位置), video_files_test(測式的影片檔案位置)。
          利用json將檔案dump至指定的json檔案位置，也就是將要使用的database，接著就可進入模型訓練的階段，只需再將json檔案載入就行了。