# DDoS-Attack-Detection-ML

This repository contains code for a DDoS attack detection system. It uses machine learning techniques to classify network traffic as DDoS or non-DDoS.

## Prerequisites

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


## Example

See the [DDoS_Detection_using_ML.ipynb](https://github.com/khangklj/DDoS-Attack-Detection-ML/tree/main/examples/DDoS_Detection_using_ML.ipynb) for more data preprocessing process detailed.

## Installation

1. Clone the repository:
```
git clone https://github.com/khangklj/DDoS-Attack-Detection-ML.git
```


2. Install the required packages:
```
pip install -r requirements.txt
```


## Usage

1. Run the main script:

```
python ddos_detection.py --classification_type [binary|multi|both] --is_saved_fig [True|False]
```

- `--classification_type`: Specifies the type of classification to perform. Options are `binary`, `multi`, or `both`. Default is `both`.
- `--is_saved_fig`: Specifies whether to save the visualized figures. Default is `False`.

2. The program will preprocess the dataset, train and test the models, and visualize the results.

## Dataset

The dataset used in this project is the [CIC-DDoS2019 Dataset]

(https://www.unb.ca/cic/datasets/ddos-2019.html)

Iman Sharafaldin, Arash Habibi Lashkari, Saqib Hakak, and Ali A. Ghorbani, "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy", IEEE 53rd International Carnahan Conference on Security Technology, Chennai, India, 2019.

(https://data.mendeley.com/datasets/ssnc74xm6r/1)

Talukder, Md Alamin; Uddin, Md Ashraf (2023), “CIC-DDoS2019 Dataset”, Mendeley Data, V1, doi: 10.17632/ssnc74xm6r.1


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
