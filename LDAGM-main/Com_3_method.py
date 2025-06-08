import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# #COMPARE 4 PHƯƠNG PHÁP
# # Nhập dữ liệu từ bảng (chỉ số trung bình, bỏ qua độ lệch chuẩn)
# data = [
#     # Dataset1
#     ['Dataset1', 'Zhang', 0.983, 0.988, 0.930, 0.941, 0.983, 0.925, 0.939],
#     ['Dataset1', 'Cải tiến 1', 0.952, 0.958, 0.758, 0.876, 0.925, 0.820, 0.869],
#     ['Dataset1', 'Cải tiến 2', 0.984, 0.989, 0.917, 0.959, 0.966, 0.950, 0.958],
#     ['Dataset1', 'Cải tiến 1+2', 0.955, 0.958, 0.777, 0.888, 0.873, 0.909, 0.890],
#     # Dataset2
#     ['Dataset2', 'Zhang', 0.953, 0.951, 0.770, 0.883, 0.915, 0.846, 0.879],
#     ['Dataset2', 'Cải tiến 1', 0.962, 0.960, 0.806, 0.903, 0.908, 0.898, 0.902],
#     ['Dataset2', 'Cải tiến 2', 0.956, 0.956, 0.798, 0.899, 0.883, 0.920, 0.901],
#     ['Dataset2', 'Cải tiến 1+2', 0.962, 0.961, 0.819, 0.909, 0.890, 0.934, 0.911],
# ]
#
# columns = ['Dataset', 'Method', 'AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
# df = pd.DataFrame(data, columns=columns)
# def plot_radar_chart(df_subset, title):
#     metrics = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
#     labels = df_subset["Method"].tolist()
#     values = df_subset[metrics].values
#
#     # Chuẩn hóa góc để vẽ tròn
#     angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
#     values = np.concatenate((values, values[:,[0]]), axis=1)
#     angles += angles[:1]
#
#     # Vẽ biểu đồ
#     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
#
#     for i in range(len(labels)):
#         ax.plot(angles, values[i], label=labels[i], linewidth=2)
#         ax.fill(angles, values[i], alpha=0.1)
#
#     ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
#     plt.title(title, size=14, fontweight='bold')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
#     plt.tight_layout()
#     plt.show()
# # Vẽ cho Dataset1
# plot_radar_chart(df[df['Dataset'] == 'Dataset1'], "So sánh mô hình trên Dataset1")
#
# # Vẽ cho Dataset2
# plot_radar_chart(df[df['Dataset'] == 'Dataset2'], "So sánh mô hình trên Dataset2")


#COMPARE ZHANG - CẢI TIẾN 2
# # Nhập dữ liệu từ bảng (chỉ số trung bình, bỏ qua độ lệch chuẩn)
# data = [
#     # Dataset1
#     ['Dataset1', 'Zhang', 0.983, 0.988, 0.930, 0.941, 0.983, 0.925, 0.939],
#     ['Dataset1', 'Cải tiến 2', 0.984, 0.989, 0.917, 0.959, 0.966, 0.950, 0.958],
#     # Dataset2
#     ['Dataset2', 'Zhang', 0.953, 0.951, 0.770, 0.883, 0.915, 0.846, 0.879],
#     ['Dataset2', 'Cải tiến 2', 0.956, 0.956, 0.798, 0.899, 0.883, 0.920, 0.901],
# ]
#
# columns = ['Dataset', 'Method', 'AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
# df = pd.DataFrame(data, columns=columns)
# def plot_radar_chart(df_subset, title):
#     metrics = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
#     labels = df_subset["Method"].tolist()
#     values = df_subset[metrics].values
#
#     # Chuẩn hóa góc để vẽ tròn
#     angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
#     values = np.concatenate((values, values[:,[0]]), axis=1)
#     angles += angles[:1]
#
#     # Vẽ biểu đồ
#     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
#
#     for i in range(len(labels)):
#         ax.plot(angles, values[i], label=labels[i], linewidth=2)
#         ax.fill(angles, values[i], alpha=0.1)
#
#     ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
#     plt.title(title, size=14, fontweight='bold')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
#     plt.tight_layout()
#     plt.show()
# # Vẽ cho Dataset1
# plot_radar_chart(df[df['Dataset'] == 'Dataset1'], "So sánh mô hình trên Dataset1")
#
# # Vẽ cho Dataset2
# plot_radar_chart(df[df['Dataset'] == 'Dataset2'], "So sánh mô hình trên Dataset2")

# #COMPARE ZHANG - CẢI TIẾN 1
# # Nhập dữ liệu từ bảng (chỉ số trung bình, bỏ qua độ lệch chuẩn)
# data = [
#     # Dataset1
#     ['Dataset1', 'Zhang', 0.983, 0.988, 0.930, 0.941, 0.983, 0.925, 0.939],
#     ['Dataset1', 'Cải tiến 1', 0.952, 0.958, 0.758, 0.876, 0.925, 0.820, 0.869],
#     # Dataset2
#     ['Dataset2', 'Zhang', 0.953, 0.951, 0.770, 0.883, 0.915, 0.846, 0.879],
#     ['Dataset2', 'Cải tiến 1', 0.962, 0.960, 0.806, 0.903, 0.908, 0.898, 0.902],
# ]
#
# columns = ['Dataset', 'Method', 'AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
# df = pd.DataFrame(data, columns=columns)
# def plot_radar_chart(df_subset, title):
#     metrics = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
#     labels = df_subset["Method"].tolist()
#     values = df_subset[metrics].values
#
#     # Chuẩn hóa góc để vẽ tròn
#     angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
#     values = np.concatenate((values, values[:,[0]]), axis=1)
#     angles += angles[:1]
#
#     # Vẽ biểu đồ
#     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
#
#     for i in range(len(labels)):
#         ax.plot(angles, values[i], label=labels[i], linewidth=2)
#         ax.fill(angles, values[i], alpha=0.1)
#
#     ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
#     plt.title(title, size=14, fontweight='bold')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
#     plt.tight_layout()
#     plt.show()
# # Vẽ cho Dataset1
# plot_radar_chart(df[df['Dataset'] == 'Dataset1'], "So sánh mô hình trên Dataset1")
#
# # Vẽ cho Dataset2
# plot_radar_chart(df[df['Dataset'] == 'Dataset2'], "So sánh mô hình trên Dataset2")

#COMPARE ZHANG - Thúy chạy lại
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ===== 1. Nhập dữ liệu =====
# data = [
#     # Dataset1
#     ['Dataset1', 'Zhang', 0.983, 0.0058, 0.988, 0.0047, 0.930, 0.0233, 0.941, 0.0122, 0.983, 0.0106, 0.925, 0.0230, 0.939, 0.0131],
#     ['Dataset1', 'Mô hình chạy lại', 0.986, 0.0025, 0.989, 0.0016, 0.886, 0.0419, 0.941, 0.0237, 0.978, 0.0144, 0.903, 0.0597, 0.938, 0.0275],
#     # Dataset2
#     ['Dataset2', 'Zhang', 0.953, 0.0053, 0.951, 0.0036, 0.770, 0.0087, 0.883, 0.0044, 0.915, 0.0054, 0.846, 0.0069, 0.879, 0.0046],
#     ['Dataset2', 'Mô hình chạy lại', 0.956, 0.0002, 0.956, 0.0003, 0.791, 0.0058, 0.895, 0.0030, 0.901, 0.0036, 0.888, 0.0110, 0.895, 0.0038]
# ]
#
# metrics = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
# columns = ['Dataset', 'Method']
# for metric in metrics:
#     columns += [metric, metric + '_std']
#
# df = pd.DataFrame(data, columns=columns)
#
# # ===== 2. Hàm vẽ biểu đồ radar =====
# def plot_radar_with_std(df_subset, title):
#     labels = df_subset['Method'].tolist()
#     angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
#     angles += angles[:1]
#
#     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
#
#     for idx, row in df_subset.iterrows():
#         values = [row[m] for m in metrics]
#         stds = [row[m + '_std'] for m in metrics]
#         values += values[:1]
#         stds += stds[:1]
#
#         ax.plot(angles, values, label=row['Method'], linewidth=2)
#         ax.fill_between(angles,
#                         np.array(values) - np.array(stds),
#                         np.array(values) + np.array(stds),
#                         alpha=0.2)
#
#     ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
#     plt.title(title, size=14, fontweight='bold')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
#     plt.tight_layout()
#     plt.show()
#
# # ===== 3. Vẽ biểu đồ cho từng dataset =====
# plot_radar_with_std(df[df['Dataset'] == 'Dataset1'], "So sánh mô hình trên Dataset1")
# plot_radar_with_std(df[df['Dataset'] == 'Dataset2'], "So sánh mô hình trên Dataset2")
# plt.savefig("dataset1_radar.png", dpi=300)


#So sánh tất cả chỉ số của tất cả các mô hình
# import seaborn as sns
#
# # Tạo bảng giá trị trung bình từ dữ liệu bạn cung cấp
# methods = ['LDAGM gốc', 'Cải tiến 1', 'Cải tiến 2', 'Cải tiến 1+2',
#            'SVM', 'RF', 'GAN', 'XGBoost']
#
# metrics = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
#
# values = [
#     [0.983, 0.988, 0.930, 0.941, 0.983, 0.925, 0.939],
#     [0.952, 0.958, 0.758, 0.876, 0.925, 0.820, 0.869],
#     [0.984, 0.989, 0.917, 0.959, 0.966, 0.950, 0.958],
#     [0.955, 0.958, 0.777, 0.888, 0.873, 0.909, 0.890],
#     [0.930, 0.920, 0.862, 0.916, 0.924, 0.783, 0.856],
#     [0.864, 0.853, 0.804, 0.884, 0.836, 0.840, 0.812],
#     [0.949, 0.952, 0.906, 0.891, 0.939, 0.848, 0.899],
#     [0.934, 0.924, 0.883, 0.905, 0.925, 0.825, 0.901]
# ]
#
# df_heat = pd.DataFrame(values, columns=metrics, index=methods)
#
# # Vẽ heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(df_heat, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
# plt.title("So sánh tổng quan các mô hình theo từng chỉ số", fontsize=14)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()



# Chỉ so sánh AUC và AUPR cho tất cả các mô hình
import matplotlib.pyplot as plt
import numpy as np

# Danh sách mô hình
models = ['LDAGM gốc', 'Cải tiến 1', 'Cải tiến 2', 'Cải tiến 1+2',
          'SVM', 'RF', 'GAN', 'XGBoost']

# AUC: [mean, std]
auc_means = [0.983, 0.952, 0.984, 0.955, 0.930, 0.864, 0.949, 0.934]
auc_stds  = [0.0058, 0.0021, 0.0057, 0.002,  0.0065, 0.0145, 0.0025, 0.0074]

# AUPR: [mean, std]
aupr_means = [0.988, 0.958, 0.989, 0.958, 0.920, 0.853, 0.952, 0.924]
aupr_stds  = [0.0047, 0.0016, 0.0039, 0.0017, 0.0136, 0.0365, 0.0069, 0.0051]

x = np.arange(len(models))  # vị trí các cột
width = 0.35  # độ rộng mỗi nhóm cột

fig, ax = plt.subplots(figsize=(12, 6))
# Cột AUC
rects1 = ax.bar(x - width/2, auc_means, width, yerr=auc_stds, label='AUC', capsize=5, color='#4C72B0')
# Cột AUPR
rects2 = ax.bar(x + width/2, aupr_means, width, yerr=aupr_stds, label='AUPR', capsize=5, color='#55A868')

# Ghi nhãn
ax.set_ylabel('Giá trị')
ax.set_title('So sánh AUC và AUPR giữa các mô hình')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylim(0.8, 1.0)
ax.legend()

# Hiển thị giá trị trên đầu cột (tùy chọn)
def autolabel(rects, errors):
    for rect, err in zip(rects, errors):
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height + err + 0.003),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1, auc_stds)
autolabel(rects2, aupr_stds)

plt.tight_layout()
plt.show()
