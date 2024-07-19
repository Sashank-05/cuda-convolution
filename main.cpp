#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QCheckBox>
#include <QtCore/QTimer>
#include <QtWidgets/QMessageBox>
#include <QtGui/QImage>
#include <QtGui/QPixmap>

// STB Image Library
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convolveImage(const unsigned char* input, unsigned char* output,
    int width, int height, int channels,
    const float* kernel, int kernelSize);

class ConvolutionGUI : public QMainWindow {
    Q_OBJECT

public:
    ConvolutionGUI(QWidget *parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("CUDA Image Convolution");
        setupUI();
    }

private:
    QLabel *imageLabel;
    QTableWidget *matrixInput;
    QPushButton *loadImageBtn;
    QPushButton *applyConvolutionBtn;
    QComboBox *matrixComboBox;
    QCheckBox *grayscaleCheckBox;
    QVBoxLayout *sidePanelLayout;
    QWidget *sidePanel;
    QImage scaledImage;
    unsigned char* inputImage = nullptr;
    int width = 0, height = 0, channels = 0;

    void setupUI() {
        QWidget *centralWidget = new QWidget(this);
        QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

        imageLabel = new QLabel(this);
        imageLabel->setScaledContents(true);
        imageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

        matrixComboBox = new QComboBox(this);
        matrixComboBox->addItem("Custom");
        matrixComboBox->addItem("Laplace");
        matrixComboBox->addItem("Sobel-X");
        matrixComboBox->addItem("Sobel-Y");
        matrixComboBox->addItem("Gaussian Blur");
        matrixComboBox->addItem("Sharpen");

        matrixInput = new QTableWidget(3, 3, this);
        matrixInput->setFixedSize(200, 200); // Fixed size for matrix boxes
        matrixInput->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        matrixInput->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {

                if (i == 1 && j == 1){
                    matrixInput->setItem(i, j, new QTableWidgetItem("1"));

                }
                else {
                matrixInput->setItem(i, j, new QTableWidgetItem("0"));
                }
            }
        }

        loadImageBtn = new QPushButton("Load Image", this);
        applyConvolutionBtn = new QPushButton("Apply Convolution", this);
        grayscaleCheckBox = new QCheckBox("Convert to Grayscale", this);

        sidePanel = new QWidget(this);
        sidePanelLayout = new QVBoxLayout(sidePanel);
        sidePanelLayout->addWidget(matrixComboBox);
        sidePanelLayout->addWidget(matrixInput);
        sidePanelLayout->addWidget(loadImageBtn);
        sidePanelLayout->addWidget(applyConvolutionBtn);
        sidePanelLayout->addWidget(grayscaleCheckBox);
        sidePanelLayout->addStretch(); // Push widgets to the top

        // Add a layout to place the side panel on the left or right
        mainLayout->addWidget(sidePanel);
        mainLayout->addWidget(imageLabel, 1); // Stretch image label

        setCentralWidget(centralWidget);

        connect(loadImageBtn, &QPushButton::clicked, this, &ConvolutionGUI::loadImage);
        connect(applyConvolutionBtn, &QPushButton::clicked, this, &ConvolutionGUI::applyConvolution);
        connect(matrixComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ConvolutionGUI::onMatrixChanged);
    }

    void loadImage() {
        QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");
        if (!fileName.isEmpty()) {
            stbi_image_free(inputImage); // Free previous image data if any

            inputImage = stbi_load(fileName.toStdString().c_str(), &width, &height, &channels, 0);
            if (!inputImage) {
                QMessageBox::warning(this, "Error", "Failed to load image.");
                return;
            }

            QImage originalImage(inputImage, width, height, width * channels, QImage::Format_RGB888);

            QSize maxSize(1200, 800);
            if (originalImage.width() > maxSize.width() || originalImage.height() > maxSize.height()) {
                scaledImage = originalImage.scaled(maxSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            } else {
                scaledImage = originalImage;
            }

            imageLabel->setPixmap(QPixmap::fromImage(scaledImage));
            updateConvolution();
        }
    }

    /*void applyConvolution() {
        if (scaledImage.isNull()) {
            QMessageBox::warning(this, "Error", "Please load an image first.");
            return;
        }

        float kernel[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                bool ok;
                float value = matrixInput->item(i, j)->text().toFloat(&ok);
                if (!ok) {
                    QMessageBox::warning(this, "Error", "Invalid kernel value. Please enter numbers only.");
                    return;
                }
                kernel[i * 3 + j] = value;
            }
        }

        if (grayscaleCheckBox->isChecked() && channels == 3) {
            convertToGrayscale(inputImage, width, height);
            channels = 1;
        }

        unsigned char* outputImage = new unsigned char[width * height * channels];

        try {
            convolveImage(inputImage, outputImage, width, height, channels, kernel, 3);
            QImage result(outputImage, width, height, width * channels, channels == 1 ? QImage::Format_Grayscale8 : QImage::Format_RGB888);
            scaledImage = result.scaled(scaledImage.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            imageLabel->setPixmap(QPixmap::fromImage(scaledImage));
        }
        catch (const std::exception& e) {
            QMessageBox::critical(this, "Error", QString("CUDA error: %1").arg(e.what()));
        }

        delete[] outputImage;
    }*/

   void applyConvolution() {
    if (scaledImage.isNull()) {
        QMessageBox::warning(this, "Error", "Please load an image first.");
        return;
    }

    float kernel[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            bool ok;
            float value = matrixInput->item(i, j)->text().toFloat(&ok);
            if (!ok) {
                QMessageBox::warning(this, "Error", "Invalid kernel value. Please enter numbers only.");
                return;
            }
            kernel[i * 3 + j] = value;
        }
    }

    
    if (grayscaleCheckBox->isChecked() && channels == 3) {
        convertToGrayscale(inputImage, width, height, channels);
        channels = 1; 
    }

    unsigned char* outputImage = new unsigned char[width * height * channels];
    
    try {
        convolveImage(inputImage, outputImage, width, height, channels, kernel, 3);
      
        QImage result(outputImage, width, height, width * channels, channels == 1 ? QImage::Format_Grayscale8 : QImage::Format_RGB888);
        
        scaledImage = result.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        imageLabel->setPixmap(QPixmap::fromImage(scaledImage));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", QString("CUDA error: %1").arg(e.what()));
    }

    delete[] outputImage;
}


    void updateConvolution() {
        QTimer::singleShot(500, this, &ConvolutionGUI::applyConvolution);
    }

    void onMatrixChanged(int index) {
        switch (index) {
        case 1: // Laplace
            setMatrixValues({0, -1, 0, -1, 4, -1, 0, -1, 0});
            break;
        case 2: // Sobel-X
            setMatrixValues({-1, 0, 1, -2, 0, 2, -1, 0, 1});
            break;
        case 3: // Sobel-Y
            setMatrixValues({-1, -2, -1, 0, 0, 0, 1, 2, 1});
            break;
        case 4: // Gaussian Blur
            setMatrixValues({1 / 16.0, 2 / 16.0, 1 / 16.0, 2 / 16.0, 4 / 16.0, 2 / 16.0, 1 / 16.0, 2 / 16.0, 1 / 16.0});
            break;
        case 5: // Sharpen
            setMatrixValues({0, -1, 0, -1, 5, -1, 0, -1, 0});
            break;
        default:
            setMatrixValues({0, 0, 0, 0, 1, 0, 0, 0, 0});
            break;
        }
    }

    void setMatrixValues(const std::vector<float>& values) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                matrixInput->item(i, j)->setText(QString::number(values[i * 3 + j]));
            }
        }
    }

    void convertToGrayscale(unsigned char* image, int width, int height, int channels) {
    if (channels == 3) {
        for (int i = 0; i < width * height; ++i) {
            unsigned char* pixel = &image[i * 3];
            unsigned char gray = static_cast<unsigned char>(0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2]);
            pixel[0] = pixel[1] = pixel[2] = gray;
        }
    }
}

};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    ConvolutionGUI window;
    window.resize(1200, 800); 
    window.show();
    return app.exec();
}

#include "main.moc"
