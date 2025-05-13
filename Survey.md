# Class Incremental Learning
Paper: Zhou, Da-Wei, Qi-Wei Wang, Zhi-Hong Qi, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. "Class-incremental learning: A survey." IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

# 1. Abstract

- Trong thực tế, các dữ liệu mới cũng như dữ liệu mới xuất hiện một cách liên tục,  đòi hỏi hệ thống học phải liên tục tiếp thu kiến thức mới.
- Khi huấn luyện trực tiếp mô hình với các mẫu từ lớp mới, một vấn đề nghiêm trọng xảy ra — mô hình có xu hướng quên nghiêm trọng các đặc điểm của các lớp trước đó, làm giảm hiệu suất một cách đáng kể.

# 2. Introduction

- Trong thế giới mở, dữ liệu huấn luyện thường có dạng dòng liên tục (streaming data). Những dữ liệu dòng này không thể được lưu trữ lâu dài do hạn chế về bộ nhớ hoặc các vấn đề về quyền riêng tư [6], khiến cho mô hình buộc phải được cập nhật dần dần chỉ với các mẫu từ lớp mới.
- Vấn đề nghiêm trọng nhất trong CIL được gọi là hiện tượng catastrophic forgetting, tức là khi tối ưu mô hình với các lớp mới, kiến thức về các lớp cũ sẽ bị xóa bỏ, dẫn đến suy giảm hiệu suất không thể phục hồi.
- Một mô hình tốt cần đạt được sự cân bằng giữa việc mô tả đặc điểm của các lớp mới và duy trì mô hình hóa đúng đắn các lớp cũ đã học. Sự đánh đổi này còn được gọi là "mâu thuẫn giữa tính ổn định và tính mềm dẻo" (stability-plasticity dilemma) trong hệ thần kinh [7], trong đó:
    - **Tính ổn định (stability)** là khả năng duy trì kiến thức đã học,
    - **Tính mềm dẻo (plasticity)** là khả năng thích nghi với các mẫu mới.
- Bên cạnh Class Incremental Learning còn 2 loại Incremental learning:
    - **Học tăng dần theo nhiệm vụ** (Task-Incremental Learning – TIL)
    - **Học tăng dần theo miền** (Domain-Incremental Learning – DIL)
- Một yếu tố quan trọng trong việc đánh giá mô hình CIL, đó là **ngân sách bộ nhớ** (memory budget).

# 3. Preliminaries

## 3.1 Problem Formulation

Học tăng dần theo lớp (Class-Incremental Learning - CIL) nhằm học từ dòng dữ liệu tiến hóa chứa các lớp mới. Giả sử có một chuỗi gồm B nhiệm vụ huấn luyện $\mathcal{D}_1,$ $\mathcal{D}_2, \cdots, \mathcal{D}_B$ **không có lớp chồng lặp**, trong đó $\mathcal{D}_b = \{(x_i^b, y_i^b)\}_{i=1}^{n_b}$ là nhiệm vụ học thứ b với $n_b$ mẫu huấn luyện. Tại đây:

- $x_i^b \in \mathbb{R}^D$ là một đầu vào thuộc lớp $y_i^b \in \mathcal{Y}_b$
- Tập nhãn giữa các nhiệm vụ không giao nhau: $\mathcal{Y}_b \cap \mathcal{Y}_{b'} = \emptyset$ với $b \ne b$

Mô hình **chỉ được truy cập dữ liệu của nhiệm vụ hiện tại** tại thời điểm huấn luyện tương ứng. Mục tiêu cuối cùng của CIL là xây dựng một mô hình phân loại bao quát **tất cả các lớp đã thấy**. Mục tiêu học là tìm một mô hình $f^*(x): \mathcal{X} \rightarrow \mathcal{Y}_{1:b}$ sao cho hàm mất mát kỳ vọng được tối thiểu:

$$
f^* = \arg\min_{f \in \mathcal{H}} \mathbb{E}_{(x,y) \sim \mathcal{D}_1 \cup \cdots \cup \mathcal{D}_b} \left[ \mathbb{I}(y \ne f(x)) \right]
$$

- $\mathcal{H}$: không gian giả thuyết
- $I(⋅)$: hàm chỉ thị, trả về 1 nếu điều kiện đúng
- $\mathcal{D}_t$: phân phối dữ liệu của nhiệm vụ t

## 3.2 Exemplars and Exemplar Set

Do mô hình **chỉ được truy cập dữ liệu hiện tại** $\mathcal{D}_b$ tại mỗi bước tăng dần, điều này giúp:

- Bảo vệ **quyền riêng tư dữ liệu người dùng**
- Giảm nhẹ **gánh nặng lưu trữ**

Tuy nhiên, nhiều phương pháp **nới lỏng hạn chế này** bằng cách **lưu lại một tập mẫu nhỏ** gọi là **Exemplar Set** để đại diện cho các lớp cũ.

### **Quản lý Exemplar Set**

Có **hai chiến lược chính** trong việc quản lý dung lượng bộ nhớ:

1. **Giữ cố định số mẫu mỗi lớp**

2. **Giữ cố định tổng số mẫu M**

# 4. Taxonomy

CIL có thể được chia thành một số loại:

- Data Replay và **data regularization**:  tập trung giải quyết CIL bằng cách sử dụng **exemplars**, hoặc là thông qua việc **lặp lại các mẫu cũ**, hoặc **dùng chúng để điều chỉnh quá trình cập nhật mô hình**.
- Dynamic Networks: mở rộng cấu trúc mạng để tăng khả năng biểu diễn
- Parameter Regularization:  điều chỉnh tham số mô hình nhằm **ngăn mô hình bị trôi** và **chống quên**.
- Knowledge Distillation: xây dựng ánh xạ giữa các mô hình tăng dần nhằm **chống hiện tượng quên**.
- Model Rectify: tập trung vào việc **giảm thiên lệch trong dự đoán** của mô hình học tăng dần.
- Template-Based Classification: nhằm biến quá trình suy luận thành **bài toán so khớp giữa truy vấn và mẫu (template matching)**.

## 4.1 Data replay

- Mô hình có thể vượt qua **hiện tượng catastrophic forgetting** bằng cách ôn lại các mẫu đã học trước đây.
- Mô hình sẽ ôn tập bằng cách lưu trữ một tập hợp exemplar và **sử dụng chúng trong quá trình cập nhật**

### Cách xây dựng tập Exemplar

- Tập hợp một tập con dữ liệu từ tập hợp gốc sao cho đa dạng nhất có thể
- Sử dụng mô hình tạo sinh để sinh ra dữ liệu huấn luyện.

### Nhận xét

- **Replay** là phương pháp đơn giản nhưng hiệu quả. Tuy nhiên, do **exemplar chỉ là tập con nhỏ**, mô hình dễ bị **overfitting** và làm giảm khả năng khái quát
- Đối với phương pháp sử dụng mô hình tạo sinh, hiệu quả phụ thuộc vào **chất lượng mẫu sinh**. Hơn nữa này cũng có thể bị **catastrophic forgetting**,

### Survey

## 4.2 Data Regularization

- Do việc học các lớp mới thường dẫn đến **hiện tượng quên thảm họa (catastrophic forgetting)** đối với các lớp cũ, nên ý tưởng trực quan là đảm bảo rằng việc tối ưu mô hình cho lớp mới **không gây hại** cho các lớp đã học trước.
- Các phương pháp **regularization** sử dụng exemplar **như một chỉ báo về mức độ quên**, giả định rằng **mất mát (loss)** trên exemplar đại diện cho các tác vụ trước. Do đó, mô hình được cập nhật sao cho gradient **căn chỉnh theo hướng bảo toàn kiến thức cũ**.
- Ngoài ra, vì cả **replay và regularization** đều **cần lưu trữ dữ liệu cũ**, nên chúng cùng gặp các vấn đề:
    - **Overfitting** do tập dữ liệu nhỏ.
    - **Mất khả năng khái quát hóa (generalization)**
    - **Vấn đề riêng tư (privacy)** nếu dữ liệu chứa thông tin nhạy cảm

### Survey

## 4.3 Dynamic Networks

- Do khả năng biểu diễn của mô hình bị giới hạn, việc học đặc trưng mới thường **ghi đè** đặc trưng cũ, dẫn đến **hiện tượng quên kiến thức đã học.**
- Giải pháp: Mạng động được thiết kế để **tự điều chỉnh khả năng biểu diễn** nhằm thích nghi với dòng dữ liệu thay đổi liên tục. Có nhiều cách mở rộng khả năng biểu diễn, được chia thành 3 nhóm:
    - Mở rộng Nơ-ron (Neuron Expansion)
    - Mở rộng Backbone (Backbone Expansion)
    - Prompt Expansion

### Survey

## 4.4 Parameter Regularization

Một hướng giải quyết là **các phương pháp regularization tham số** (parameter regularization), dựa trên ý tưởng rằng:

- Không phải **tham số nào cũng quan trọng như nhau**.
- Do đó, cần **ước lượng tầm quan trọng của từng tham số**, và **cố định** các tham số quan trọng để giữ lại kiến thức cũ.

Mặc dù **parameter regularization** và **data regularization** đều dùng regularization để chống quên, nhưng:

- **Data regularization**: dựa vào **tập mẫu đại diện (exemplar)** để điều hướng tối ưu.
- **Parameter regularization**: dựa vào **tầm quan trọng từng tham số** để tạo ràng buộc.

### Survey

## 4.5 Knowledge Distillation

- Một cách tiếp cận trực quan là sử dụng *knowledge distillation – KD*, khái niệm được đề xuất trong [178]. KD cho phép **truyền tải kiến thức** từ mô hình giáo viên sang mô hình học sinh, nhờ đó giúp mô hình mới không bị quên kiến thức cũ.
- Các phương pháp KD được chia làm 3 nhóm:
    1. **Logit distillation** – chưng cất xác suất đầu ra
    2. **Feature distillation** – chưng cất đặc trưng trung gian
    3. **Relational distillation** – chưng cất mối quan hệ giữa các mẫu
- **Hạn chế**:
    - **Khó điều chỉnh** chính xác mức cân bằng giữa học cái mới (plasticity) và giữ cái cũ (stability).
    - Nếu đặt trọng số quá lớn cho KD → **giảm khả năng học mới**.
    - Nếu quá thấp → **gây quên thảm họa**.

### Survey

## 4.6 Model Rectify

Nếuthu thập toàn bộ tập dữ liệu huấn luyện ngay từ đầu và huấn luyện mô hình với nhiều epoch (lượt), thì mô hình sẽ không bị quên và sẽ hoạt động tốt với tất cả các lớp. Cách huấn luyện lý tưởng này được gọi là **"Oracle" (Mô hình lý tưởng)**, được xem là **giới hạn trên (upper bound)** của CIL.

Tuy nhiên, trong CIL, do dữ liệu được cung cấp dần dần, mô hình sẽ bị **quên thảm khốc (catastrophic forgetting)**. Nhiều phương pháp đã cố gắng xác định **các hành vi bất thường** của mô hình CIL và **chỉnh sửa chúng** sao cho mô hình gần giống với Oracle. Những hành vi bất thường này có thể là:

- Logits đầu ra bị lệch
- Trọng số bộ phân loại (classifier weights) sai lệch
- Biểu diễn đặc trưng (feature embeddings) bị trôi

### Survey

## 4.7 Template-Based Classification

Ý tưởng là: nếu chúng ta có thể xây dựng một **mẫu (template)** cho mỗi lớp, ta có thể phân loại bằng cách **so khớp đầu vào với mẫu gần nhất**. Một phương pháp tiêu biểu là dùng **prototype (đại diện)** cho mỗi lớp – là **trung bình** các vector đặc trưng của các mẫu trong lớp đó (có nguồn gốc từ khoa học nhận thức).

Công thức trích xuất prototype cho lớp i:

$$
p_i = \frac{1}{N} \sum_{j=1}^{|D_b|} I(y_j = i)\phi(x_j)
$$

Dự đoán nhãn:

$$
y^* = \arg\min_y \|\phi(x) - p_y\|
$$

Phương pháp này tránh được sai lệch do trọng số của lớp fully-connected.

**Ưu điểm:**

- Giảm **lệch mô hình** trong CIL bằng cách so khớp trực tiếp trong không gian đặc trưng.
- Nếu có mô hình tiền huấn luyện tốt, chỉ cần trích xuất đặc trưng mà không cần huấn luyện lại → không bị quên.

**Nhược điểm:**

- Khi **không có exemplar**, việc tái tính prototype là **không thể**, cần kỹ thuật phức tạp để bù trôi ngữ nghĩa.
- **Đóng băng backbone** sẽ làm **giảm khả năng thích ứng** với dữ liệu mới, đặc biệt khi có **sự khác biệt miền (domain gap)** lớn giữa dữ liệu gốc và dữ liệu mới.

**Giải pháp thay thế:**

- **Energy-based models**: dùng năng lượng thay vì xác suất để đo độ tương thích với từng lớp → tiết kiệm chi phí hơn so với mô hình sinh.

### Survey

# 5. Thực nghiệm

## 5.1 Dữ liệu

- Một số bộ dữ liệu phổ biến trong thực nghiệm CIL là: CIFAR100, ImageNet100/1000, MNIST, CUB200, miniImageNet.
- Để chuẩn bị dữ liệu cho CIL, có **hai cách chia các lớp thành các giai đoạn tăng dần (incremental stages):**
    - **Huấn luyện từ đầu (Train from Scratch – TFS):**
        - Chia đều toàn bộ các lớp thành từng giai đoạn.
        - Ví dụ: nếu có tổng cộng C lớp và chia thành B giai đoạn, thì mỗi giai đoạn chứa **C/B lớp** để huấn luyện.
    - **Huấn luyện từ một nửa (Train from Half – TFH):**
        - Gán **một nửa số lớp** cho giai đoạn đầu tiên.
        - Phần còn lại được chia đều cho các giai đoạn tiếp theo.
        - Cụ thể: giai đoạn đầu có **C/2 lớp**, còn mỗi giai đoạn sau có **C/2(B−1) lớp**.

## 5.2 Độ đo và đánh giá

Để đánh giá một thuật toán CIL, có một số chỉ số như sau:

- Độ chính xác của mỗi task
- Độ chính xác của task cuối cùng
- Độ chính xác trung bình của các task
- Ngoài ra còn có thể đo:
    - **Forgetting:** mức độ mô hình **quên** kiến thức cũ.
    - **Intransigence:** mức độ mô hình **khó học** cái mới.

## 5.3 Cài đặt

### Backbone:

- CIFAR100: **ResNet32**
- ImageNet: **ResNet18**
- ViT: ConViT

### Hyperparameters:

- SGD, LR ban đầu = 0.1, momentum = 0.9
- 170 epochs
- Batch size = 128
- LR giảm 0.1 tại epoch 80 và 120
