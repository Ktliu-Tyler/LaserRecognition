# **Laser_recognition模型製作與影像前置處理**
### **Author: Hdingzp4  Tylerj86**
* **影像資料抓取**：<br>
    * 使用colab:<br>
        ```python
        from google.colab import drive
        drive.mount('/content/gdrive')
        ```
        ```python
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        import os

        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        ```
        我們的影像檔案都優先存儲於雲端硬碟中的1sec_video資料夾中，由於是使用colab進行編寫，我們引入google.colab.drive將colab掛載至雲端硬碟上以取得data並利於建立database。<br>
    * 使用jupyter:<br>
        只須直接使用本地影片路徑即可。

    * 影像處理:<br>
      * Video_process_tool:<br>
        ```python
        class Video_process_tool:
        def __init__(self):
            pass
        def get_mask(self, img):
            mask = img[img.shape[0] // 2 - 175: img.shape[0] // 2 + 125,
                    img.shape[1] // 2 - 125: img.shape[1] // 2 + 125]
            return mask
        def resize_img(self, img):
            return cv2.resize(img,None,fx=0.3,fy=0.3)

        def gray_img(self, img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ```
          為符合CNN LSTM模型所需訓練的格式，首先我們利用opencv-python模組進行影片的前置處理。
          我們建立了名為Video_process_tool的Class以利處理影像，於其中建立了get_mask resize_img及gray_img三種函式。
        * get_mask:<br>
            針對拍攝的影像設定一個固定的mask以清晰抓取的震動並切割。
        * resize_img:<br>
            將圖片縮放為63x75的影像。
        * gray_img:<br>
            利用opencv的cv2.cvtColor模組先將圖片轉成灰階。
      * 建立frames_extraction函式:<br>
        ```python
        def frames_extraction(video_path):

            # Declare a list to store video frames.
            frames_list = []

            # Read the Video File using the VideoCapture object.
            video_reader = cv2.VideoCapture(video_path)

            # Get the total number of frames in the video.
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the the interval after which frames will be added to the list.
            skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

            # Iterate through the Video Frames.
            for frame_counter in range(SEQUENCE_LENGTH):

                # Set the current frame position of the video.
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                # Reading the frame from the video.
                success, frame = video_reader.read()


                # Check if Video frame is not successfully read then break the loop
                if not success:
                    break

                frame = vid.resize_img(frame)

                frame = vid.get_mask(frame)
                # print(frame.shape)
                # frame = vid.gray_img(frame)

                # Resize the Frame to fixed height and width.
                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
                normalized_frame = resized_frame / 255

                # Append the normalized frame into the frames list
                frames_list.append(normalized_frame)

            # Release the VideoCapture object.
            video_reader.release()

            # Return the frames list.
            return frames_list
        ```
        使用cv2抓取檔案中的影像，檢測影片抓取的幀數，加以切割並分配長度設定為15幀的影像。再將影像經前述的Video_proccess_tool處理後，對於每個pixel除以255，也就是使其成為介於0到1之間的數值，以方便後續進行卷積或是運算，最後將每幀圖片加入陣列中回傳。
      * 建立create_database函式:<br>
        ```python
        # 創建訓練資料集函式
        def create_dataset():
            '''
            This function will extract the data of the selected classes and create the required dataset.
            Returns:
                features:          A list containing the extracted frames of the videos.
                labels:            A list containing the indexes of the classes associated with the videos.
                video_files_paths: A list containing the paths of the videos in the disk.
            '''

            # Declared Empty Lists to store the features, labels and video file path values.
            features = []
            labels = []
            video_files_paths = []

            # Iterating through all the classes mentioned in the classes list
            for class_index, class_name in enumerate(CLASSES_LIST):

                # Get the list of video files present in the specific class name directory.
                files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

                # Display the name of the class whose data is being extracted.
                print(f'Extracting Data of Class: {class_name}, Total File Num: {len(files_list)}')

                # Iterate through all the files present in the files list.
                for num, file_name in enumerate(files_list):

                    print(f'Processing {num+1} Data')
                    # Get the complete video path.
                    video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

                    # Extract the frames of the video file.
                    frames = frames_extraction(video_file_path)

                    # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
                    # So ignore the vides having frames less than the SEQUENCE_LENGTH.
                    if len(frames) == SEQUENCE_LENGTH:

                        # Append the data to their repective lists.
                        features.append(frames)
                        labels.append(class_index)
                        video_files_paths.append(video_file_path)

            # Converting the list to numpy arrays
            features = np.asarray(features)
            labels = np.array(labels)

            # Return the frames, class index, and video file path.
            return features, labels, video_files_paths
        ```
        走訪資料夾中的所有影像，調用frames_extraction並取得其回傳的影祥資料加入到同樣標籤的陣列中，分成相對應的features和Labels陣列分別代表該影片的data和其對應的標籤。

      * 資料拆分與下載:<br>
        ```python
        # 將資料集拆分為訓練和測試資料集
        features_train, features_test, labels_train, labels_test, video_files_train, video_files_test = train_test_split(features, one_hot_encoded_labels, video_files_paths, random_state = seed_constant, train_size=0.8)
        ```
        使用sklearn對處理好的features和Labels，進行拆分，分成features_train(用於訓練的資料), features_test(用於測試的資料), labels_train(用於訓練的標籤), labels_test(用於測試的標籤), video_files_train(訓練的影片檔案位置), video_files_test(測式的影片檔案位置)。
        利用json將檔案dump至指定的json檔案位置，也就是將要使用的database，接著就可進入模型訓練的階段，只需再將json檔案載入就行了。