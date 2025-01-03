## The code for Deep Regularized Waveform Learning for Beam Prediction With Limited Samples in Non-Cooperative mmWave Systems

# Paper
H. Huang, G. Gui, H. Gacanin, C. Yuen, H. Sari and F. Adachi, "Deep Regularized Waveform Learning for Beam Prediction With Limited Samples in Non-Cooperative mmWave Systems," IEEE Transactions on Vehicular Technology, vol. 72, no. 7, pp. 9614-9619, July 2023.

# Abstract
Millimeter wave (mmWave) systems need beam management to establish and maintain reliable links. This complex and time-consuming process seriously affects communication efficiency. Benefiting from data-driven technology in deep learning, the beam can be predicted from the waveform without coordination between transceivers. By passively listening enough waveforms that are transmitted from the base station (BS) to other receivers, the BS can predict which beam is suitable for transmitting in the downlink. However, training such a waveform learning neural network usually requires a large number of labeled training samples. This is a huge challenge, because it is difficult for the receiver to get the precise signal parameters from the transmitter in advance in the non-cooperative mmWave system. As a result, the limited samples may cause overfitting and seriously restrict the performance. Although the data augmentation technology can improve the performance under limited samples, existing data augmentation methods are mostly based on strong prior knowledge which cannot further exploit the potential characteristics of data in the real environment. This paper proposes a mixed regularization training method for training the beam prediction neural network under limited training samples. Specifically, data augmentation is implemented in the data pre-processing procedure with prior knowledge and then the signal splicing strategy is proposed in the training procedure. In order to mine the time correlation characteristics of signals, the cyclic time shift (CTS) based data augmentation method is proposed in the data augmentation step. The simulation results show that our proposed deep regularized waveform learning method needs less training samples under the same performance. Moreover, the proposed method can achieve best performance compared with existing data augmentation methods.

# Results (Code of Proposed 2)
![image](https://github.com/BeechburgPieStar/beam-prediction/assets/107237593/65ee9f54-d675-4c58-8b4d-6239fcff80a3)

# License / 许可证

本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.

