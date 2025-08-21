import streamlit as st
import numpy as np
import cv2
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from evaluator import BinaryImageEvaluatorNp

st.set_page_config(page_title="MatVision Image Evaluator", layout="wide")
st.title("📊 MatVision Similarity Evaluator")

# ---------------- 参数设置 ----------------
threshold = st.sidebar.slider("Binarization Threshold", 0.0, 1.0, 0.5, 0.01)
metrics = st.sidebar.multiselect(
    "Select Metrics",
    ['iou','dice','hd','ssim','porosity','pore_size','two_point'],
    default=['iou','dice','ssim','pore_size','two_point']
)

# ---------------- 文件上传 ----------------
st.subheader("Upload Real Images")
real_files = st.file_uploader(
    "Select multiple files: .npy, .h5, .jpg, .jpeg, .png, .tif, .tiff",
    type=['npy','h5','jpg','jpeg','png','tif','tiff'],
    accept_multiple_files=True,
    key="real_files_uploader"
)

st.subheader("Upload Generated Images")
fake_files = st.file_uploader(
    "Select multiple files: .npy, .h5, .jpg, .jpeg, .png, .tif, .tiff",
    type=['npy','h5','jpg','jpeg','png','tif','tiff'],
    accept_multiple_files=True,
    key="fake_files_uploader"
)

# ---------------- 数据读取 ----------------
def load_data(file):
    if file.name.endswith('.npy'):
        data = np.load(file)
    elif file.name.endswith('.h5'):
        with h5py.File(file,'r') as f:
            key = list(f.keys())[0]
            data = f[key][:]
    else:  # 图片格式
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        data = img.astype(np.float32)/255.0
    if data.ndim > 2:
        data = data[0]
    return data.astype(np.float32)

def load_multiple(files):
    arrays = [load_data(f) for f in files]
    if not arrays:
        return np.empty((0,1,1))
    stacked = np.stack(arrays, axis=0)
    return stacked

# ---------------- 裁剪到统一最大尺寸 ----------------
def crop_to_max_size(real_images, fake_images):
    all_images = np.concatenate([real_images, fake_images], axis=0)
    max_h = max(img.shape[0] for img in all_images)
    max_w = max(img.shape[1] for img in all_images)

    def pad_image(img, target_h, target_w):
        h, w = img.shape
        pad_top = (target_h - h)//2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w)//2
        pad_right = target_w - w - pad_left
        return np.pad(img, ((pad_top,pad_bottom),(pad_left,pad_right)), mode='constant', constant_values=0)

    real_cropped = np.stack([pad_image(img, max_h, max_w) for img in real_images], axis=0)
    fake_cropped = np.stack([pad_image(img, max_h, max_w) for img in fake_images], axis=0)
    return real_cropped, fake_cropped

# ---------------- 指标说明 ----------------
metrics_info = {
    "IoU": "Intersection over Union: measures overlap between real and generated pores.",
    "Dice": "Dice coefficient: 2*|A∩B| / (|A|+|B|).",
    "HD": "Hausdorff Distance: maximum distance of a point in one image to the other.",
    "SSIM": "Structural Similarity Index: measures similarity of two images.",
    "Porosity": "Pore area divided by total area.",
}
st.info("💡 **指标说明：**")
for metric, description in metrics_info.items():
    st.write(f"**{metric}:** {description}")


# ---------------- 两点统计字典转DataFrame ----------------
def two_point_dict_to_df(tp_dict):
    rows = []
    for (dx, dy), metrics in tp_dict.items():
        distance = np.sqrt(dx**2 + dy**2)
        row = {'displacement': distance}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------- 评估 ----------------
if st.button("Run Evaluation"):
    if not real_files or not fake_files:
        st.warning("Please upload both real and generated images!")
    elif len(real_files) != len(fake_files):
        st.warning(f"Number of real images ({len(real_files)}) and generated images ({len(fake_files)}) must be the same!")
    else:
        with st.spinner("Loading images..."):
            real_images = load_multiple(real_files)
            fake_images = load_multiple(fake_files)

        # 裁剪
        real_images, fake_images = crop_to_max_size(real_images, fake_images)
        st.success(f"Loaded {real_images.shape[0]} real images and {fake_images.shape[0]} generated images")

        evaluator = BinaryImageEvaluatorNp(threshold=threshold)
        with st.spinner("Evaluating..."):
            results = evaluator.evaluate(real_images[:,np.newaxis,...], fake_images[:,np.newaxis,...], metrics=metrics)

        st.success("Evaluation Completed ✅")

        # ---------------- 指标表格 ----------------
        table_metrics = ['iou','dice','hd','ssim','porosity_real','porosity_fake']
        metrics_data = []
        for metric in table_metrics:
            if metric in results:
                name = metric.replace('_',' ').capitalize() if metric.startswith('porosity') else metric.upper()
                metrics_data.append({'metric':name,'value':results[metric]})
        if metrics_data:
            st.subheader("Evaluation Metrics")
            df_metrics = pd.DataFrame(metrics_data)
            for row in df_metrics.itertuples():
                help_text = metrics_info.get(row.metric,"")
                st.metric(label=row.metric, value=f"{row.value:.4f}", help=help_text)

        # ---------------- 孔径分布 ----------------
        if 'pore_size_real' in results and 'pore_size_fake' in results:
            st.subheader("Pore Size Distribution")
            df_pore = pd.DataFrame({
                'Diameter': results['pore_size_real']['all_diameters']+results['pore_size_fake']['all_diameters'],
                'Type': ['Real']*len(results['pore_size_real']['all_diameters']) + ['Generated']*len(results['pore_size_fake']['all_diameters'])
            })
            # 显示 DataFrame
            st.dataframe(df_pore.head(20))
            # 绘图
            fig, axes = plt.subplots(1,2,figsize=(14,5))
            sns.histplot(data=df_pore,x='Diameter',hue='Type',stat='density',bins=20,alpha=0.5,kde=True, ax=axes[0])
            axes[0].set_title("Pore Size Histogram")
            sns.violinplot(data=df_pore,x='Type',y='Diameter',inner='quartile',cut=0, ax=axes[1])
            axes[1].set_title("Pore Size Violin Plot")
            plt.tight_layout()
            st.pyplot(fig)

        # ---------------- 两点统计 ----------------
        if 'two_point_real' in results and 'two_point_fake' in results:
            st.subheader("Two-Point Statistics")
            df_real = two_point_dict_to_df(results['two_point_real'])
            df_fake = two_point_dict_to_df(results['two_point_fake'])
            df_real['Type'] = 'Real'
            df_fake['Type'] = 'Generated'
            df_tp = pd.concat([df_real, df_fake], axis=0)
            st.dataframe(df_tp.head(20))

            # 散点图
            fig, ax = plt.subplots(figsize=(10,6))
            metrics_list = ['P00','P11','P01','P10','P_connect']
            colors = ['r','g','b','c','m']
            for i, metric in enumerate(metrics_list):
                ax.scatter(df_tp['displacement'], df_tp[metric], alpha=0.6, label=metric, color=colors[i])
            ax.set_xlabel('Displacement Distance (pixels)')
            ax.set_ylabel('Probability')
            ax.set_title('Two-Point Statistics Scatter Plot')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # ---------------- Excel 分页导出 ----------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if metrics_data:
                df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            if 'pore_size_real' in results and 'pore_size_fake' in results:
                df_pore.to_excel(writer, sheet_name='PoreSize', index=False)
            if 'two_point_real' in results and 'two_point_fake' in results:
                df_tp.to_excel(writer, sheet_name='TwoPoint', index=False)
        output.seek(0)
        st.download_button(
            label="Download Evaluation Results (Excel)",
            data=output.getvalue(),
            file_name="evaluation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
