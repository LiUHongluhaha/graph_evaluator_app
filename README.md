# MatVision 材料微观影像视觉评估平台

这是一个基于 [Streamlit](https://streamlit.io/) 的材料微观影像评估工具，用于比较真实图像（Real）与生成图像（Generated）的孔隙结构和视觉相似度。应用支持批量上传图像，计算多种评价指标，并将结果导出为 Excel 文件。

## 功能特性

- 支持批量上传真实图像和生成图像
- 支持以下输入格式：
  - `.npy`
  - `.h5`
  - `.jpg` / `.jpeg`
  - `.png`
  - `.tif` / `.tiff`
- 自动将不同尺寸的图像填充到统一尺寸
- 可配置二值化阈值
- 支持以下评价指标：
  - **IoU**：交并比
  - **Dice**：Dice 相似系数
  - **HD**：Hausdorff Distance，豪斯多夫距离
  - **SSIM**：结构相似性
  - **Porosity**：孔隙率
  - **Pore Size**：孔径分布
  - **Two-Point Statistics**：两点统计信息和连接概率
- 可视化孔径分布直方图、小提琴图和两点统计散点图
- 支持将评价指标、孔径分布和两点统计结果导出为 Excel

## 环境要求

- Python 3.11 或更高版本
- 建议使用虚拟环境隔离项目依赖

## 安装

```bash
git clone https://github.com/LiUHongluhaha/graph_evaluator_app.git
cd graph_evaluator_app

python -m venv .venv
```

激活虚拟环境：

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 启动应用

```bash
streamlit run app.py
```

启动后，在浏览器中打开 Streamlit 输出的地址，默认通常为：

```text
http://localhost:8501
```

## 使用方法

1. 在左侧设置 **Binarization Threshold**，默认值为 `0.5`。
2. 在 **Select Metrics** 中选择要计算的评价指标。
3. 在 **Upload Real Images** 中上传真实图像。
4. 在 **Upload Generated Images** 中上传生成图像。
5. 确保两组图像数量相同。
6. 点击 **Run Evaluation** 开始评估。
7. 查看指标、孔径分布和两点统计可视化结果。
8. 点击 **Download Evaluation Results (Excel)** 下载结果。

真实图像和生成图像会按照上传列表中的顺序进行配对比较。建议两组文件使用一致的命名规则和排序方式。

## 输入数据说明

### 图像文件

常见图像格式会以灰度模式读取，并归一化到 `[0, 1]` 范围。多通道图像在读取时会转换为灰度图。

### NumPy 文件

`.npy` 文件应包含二维图像数组，或包含可取出二维图像的数组。数组会转换为 `float32`。

### HDF5 文件

`.h5` 文件默认读取文件中的第一个数据集。请确保该数据集包含图像数据。

### 图像尺寸

当输入图像尺寸不一致时，程序会以所有图像中的最大高度和最大宽度为目标尺寸，并使用 0 值在四周进行居中填充。

## 项目结构

```text
.
├── app.py                    # Streamlit 应用界面、数据读取和结果展示
├── evaluator.py              # 图像评价指标和统计方法实现
├── requirements.txt          # Python 依赖
├── .devcontainer/
│   └── devcontainer.json     # VS Code / GitHub Codespaces 开发容器配置
└── LICENSE                   # 项目许可证
```

## 开发容器

项目包含 `.devcontainer/devcontainer.json`，可使用 VS Code Dev Containers 或 GitHub Codespaces 打开。容器会安装 `requirements.txt` 中的依赖，并自动启动 Streamlit 服务。

## 注意事项

- 真实图像和生成图像的数量必须相同。
- 输入文件应包含有效的二维图像数据。
- `HD` 在任意一方不存在前景像素时可能返回 `inf`。
- 两点统计默认遍历 `[-50, 50]` 范围内的位移，较大的图像批次可能需要较长计算时间。
- `SSIM` 适用于归一化到 `[0, 1]` 范围的图像。

## 许可证

本项目遵循仓库中的 [LICENSE](LICENSE) 文件。
