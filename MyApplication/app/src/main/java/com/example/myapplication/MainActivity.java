package com.example.myapplication;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.LinearLayout;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.example.myapplication.ml.Dense;
import com.example.myapplication.ml.Effb0Quant;
import android.graphics.Typeface;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery, newPrediction;
    RadioButton selectDense, selectEffb0Quant;
    ImageView imageView;
    TextView modelTextView, classTextView, confidenceTextView, executionTimeTextView;
    LinearLayout selectionLayout;
    int imageSize = 224;
    boolean useDenseModel = true; // By default, use the Dense model

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button_camera);
        gallery = findViewById(R.id.button_gallery);
        newPrediction = findViewById(R.id.button_new_prediction);
        selectDense = findViewById(R.id.selectDense);
        selectEffb0Quant = findViewById(R.id.selectEffb0Quant);
        imageView = findViewById(R.id.imageView);
        selectionLayout = findViewById(R.id.selectionLayout);
        modelTextView = findViewById(R.id.modelTextView);
        classTextView = findViewById(R.id.classTextView);
        confidenceTextView = findViewById(R.id.confidenceTextView);
        executionTimeTextView = findViewById(R.id.executionTimeTextView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 1);
            }
        });

        selectDense.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                useDenseModel = true;
            }
        });

        selectEffb0Quant.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                useDenseModel = false;
            }
        });

        newPrediction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showSelectionLayout();
            }
        });
    }

    private void showSelectionLayout() {
        selectionLayout.setVisibility(View.VISIBLE);
        newPrediction.setVisibility(View.GONE);
        imageView.setVisibility(View.GONE);
        modelTextView.setVisibility(View.GONE);
        classTextView.setVisibility(View.GONE);
        confidenceTextView.setVisibility(View.GONE);
        executionTimeTextView.setVisibility(View.GONE);
    }

    private void hideSelectionLayout() {
        selectionLayout.setVisibility(View.GONE);
        newPrediction.setVisibility(View.VISIBLE);
        imageView.setVisibility(View.VISIBLE);
        modelTextView.setVisibility(View.VISIBLE);
        classTextView.setVisibility(View.VISIBLE);
        confidenceTextView.setVisibility(View.VISIBLE);
        executionTimeTextView.setVisibility(View.VISIBLE);
    }

    public void classifyImage(Bitmap image) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[imageSize * imageSize];
        image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
        int pixel = 0;

        // Normalize image pixels
        for (int i = 0; i < imageSize; i++) {
            for (int j = 0; j < imageSize; j++) {
                int val = intValues[pixel++]; // RGB
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.f);
                byteBuffer.putFloat((val & 0xFF) / 255.f);
            }
        }

        hideSelectionLayout(); // Ascunde opțiunile de selecție a modelului

        if (useDenseModel) {
            classifyImageWithDense(byteBuffer);
        } else {
            classifyImageWithEffb0Quant(byteBuffer);
        }
    }



    private void classifyImageWithDense(ByteBuffer byteBuffer) {
        try {
            Dense model = Dense.newInstance(getApplicationContext());

            // Start timing
            long startTime = System.currentTimeMillis();

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Dense.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();

            // Find the index of the class with the highest confidence
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Audi A1", "Audi A3", "Audi A4L", "Audi A5 Convertible", "Audi A5 Coupe", "Audi A5 Hatchback", "Audi A6L", "Audi A7", "Audi A8L", "Audi Q3", "Audi Q5", "Audi Q7", "Audi S5", "Audi TT", "Audi TTS", "BMW M5", "BMW Series 1", "BMW Series 3 Coupe", "BMW Series 3 Sedan", "BMW Series 5", "BMW Series 5 GT", "BMW Series 6", "BMW Series 7", "BMW X1", "BMW X3", "BMW X5", "BMW X6", "BYD F0", "BYD F3R", "BYD L3", "BYD S6", "Baic E150", "Baojun 630", "Brilliance FRV", "Brilliance FSV", "Brilliance M2", "Buick Encore", "Buick Excelle", "Buick Excelle GT", "Buick Excelle XT", "Buick GL8", "Buick GL8 II", "Buick Lacrosse", "Buick Regal", "Buick Regal GS", "Cadillac CTS", "Cadillac SRX", "Cadillac Seville", "Cadillac XTS", "Changan Alsvin", "Changan Benni Mini", "Changan CX20", "Changan Eado", "Changan S35", "Chery A3", "Chery E5", "Chery Tiggo 3", "Chevrolet Aveo", "Chevrolet Camaro", "Chevrolet Captiva", "Chevrolet Cruze", "Chevrolet Epica", "Chevrolet Lova", "Chevrolet Malibu", "Chevrolet Sail", "Chevrolet Spark", "Chrysler 300C", "Citroen C-Elysee", "Citroen C-Quatre Hatchback", "Citroen C-Quatre Sedan", "Citroen C4L", "Citroen C5", "Citroen DS3", "Citroen DS5", "Dongfeng H30", "Dongfeng Joyear", "Dongfeng S30", "Dongfeng Succe", "Emgrand EC7-RV", "Emgrand EC715", "FAW Bestune B50", "FAW Besturn B70", "FAW Besturn B90", "FAW N5", "Fiat 500", "Fiat Bravo", "Ford Ecosport", "Ford Fiesta Hatchback", "Ford Fiesta Sedan", "Ford Focus Hatchback", "Ford Focus Sedan", "Ford Mondeo", "Ford Mustang", "GAC Trumpchi GS5", "Geely", "Great Wall Haval M4", "Great Wall Hover H3", "Great Wall Voleex C50", "Great Wall Voleex V80", "Haima 2", "Haima 3", "Haima 7", "Haima Freema", "Honda Accord", "Honda CR-V", "Honda Civic", "Honda Crosstour", "Hyundai Santa Fe", "Infinity Q70L", "Infinity QX50", "Infinity QX70", "JAC A13", "JAC B15", "JAC B18", "Jaguar XF", "Jaguar XJL", "Jeep Compass", "Jeep Patriot", "Jeep Wrangler", "Kia Cadenza", "Kia Forte", "Kia K5", "Kia Koup", "Kia Sorento", "Kia Soul", "Kia Sportage", "Kia Sportage R", "Lamborghini Gallardo", "Land Rover Discovery", "Land Rover Range Rover Evoque", "Land Rover Range Rover Sport", "Land Rover Range Rover Vogue", "Landwind X8", "Lexus CT200h", "Lexus GS", "Lexus GS h", "Lexus IS", "Lexus RX", "Lifan 320", "MG3", "MG6", "Mazda 2", "Mazda 3 Hatchback", "Mazda 3 Sedan", "Mazda 5", "Mazda 6 Atenza", "Mazda 6 GG1", "Mazda 6 GH", "Mazda CX-5", "Mercedes-AMG C63", "Mercedes-Benz C-Class Hatchback", "Mercedes-Benz C-Class Sedan", "Mercedes-Benz E-Class Convertible", "Mercedes-Benz E-Class Coupe", "Mercedes-Benz E-Class Sedan", "Mercedes-Benz GLK-Class", "Mercedes-Benz R-Class", "Mercedes-Benz S-Class", "Mercedes-Benz SLK-Class", "Mini Cooper", "Mini Cooper Clubman", "Mini Cooper Countryman", "Mini Cooper Paceman", "Mitsubishi Lancer", "Mitsubishi Lancer EX", "Mitsubishi Outlander", "Mitsubishi Pajero", "Nissan Bluebird Sylphy", "Nissan GT-R", "Nissan NV200", "Nissan Qashqai", "Nissan Sunny", "Nissan Teana", "Opel Antara", "Opel Astra GTC", "Peugeot 207 Hatchback", "Peugeot 207 Sedan", "Peugeot 307", "Peugeot 308", "Peugeot 408", "Peugeot 508", "Peugeot RCZ", "Porsche 911", "Porsche Cayenne", "Porsche Cayman", "Porsche Panamera", "Renault Koleos", "Riich G5", "Roewe 350", "Roewe 550", "Skoda Fabia", "Skoda Octavia", "Skoda Rapid", "Skoda Superb", "Subaru Impreza", "Suzuki Alto", "Suzuki Kizashi", "Suzuki SX4", "Suzuki Swift", "Suzuki X5", "Toyota Camry", "Toyota Corolla", "Toyota Crown", "Toyota EZ", "Toyota GT86", "Toyota Land Cruiser Prado", "Toyota RAV4", "Toyota Reiz", "Toyota Vios", "Toyota Yaris", "Volkswagen Bora", "Volkswagen CC", "Volkswagen Dune", "Volkswagen Eos", "Volkswagen GTI", "Volkswagen Golf", "Volkswagen Jetta", "Volkswagen Lavida", "Volkswagen Magotan", "Volkswagen Multivan", "Volkswagen Passat", "Volkswagen Phaeton", "Volkswagen Polo", "Volkswagen Sagitar", "Volkswagen Scirocco", "Volkswagen Tiguan", "Volkswagen Touareg", "Volkswagen Touran", "Volvo C30", "Volvo C70", "Volvo S40", "Volvo S60", "Volvo S60L", "Volvo S80L", "Volvo V40", "Volvo V60", "Volvo XC60", "Volvo XC90", "Wuling Hongguang"};

            // End timing
            long endTime = System.currentTimeMillis();
            long executionTime = endTime - startTime;

            modelTextView.setText(String.format("Model ales: DenseNet"));
            modelTextView.setTextSize(30); // Increase text size
            modelTextView.setTypeface(null, Typeface.BOLD); // Set text to bold

            classTextView.setText(String.format("Clasa prezisă: %s", classes[maxPos]));
            classTextView.setTextSize(24); // Increase text size
            classTextView.setTypeface(null, Typeface.ITALIC); // Set text to bold and italic

            confidenceTextView.setText(String.format("Precizie: %.2f%%", maxConfidence * 100));
            confidenceTextView.setTextSize(24); // Increase text size
            confidenceTextView.setTypeface(null, Typeface.ITALIC); // Set text to italic

            executionTimeTextView.setText(String.format("Latență: %d ms", executionTime));
            executionTimeTextView.setTextSize(24); // Increase text size
            executionTimeTextView.setTypeface(null, Typeface.ITALIC); // Set text to bold

            // Make the TextViews visible
            modelTextView.setVisibility(View.VISIBLE);
            classTextView.setVisibility(View.VISIBLE);
            confidenceTextView.setVisibility(View.VISIBLE);
            executionTimeTextView.setVisibility(View.VISIBLE);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void classifyImageWithEffb0Quant(ByteBuffer byteBuffer) {
        try {
            Effb0Quant model = Effb0Quant.newInstance(getApplicationContext());

            // Start timing
            long startTime = System.currentTimeMillis();

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Effb0Quant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();

            // Find the index of the class with the highest confidence
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Audi A1", "Audi A3", "Audi A4L", "Audi A5 Convertible", "Audi A5 Coupe", "Audi A5 Hatchback", "Audi A6L", "Audi A7", "Audi A8L", "Audi Q3", "Audi Q5", "Audi Q7", "Audi S5", "Audi TT", "Audi TTS", "BMW M5", "BMW Series 1", "BMW Series 3 Coupe", "BMW Series 3 Sedan", "BMW Series 5", "BMW Series 5 GT", "BMW Series 6", "BMW Series 7", "BMW X1", "BMW X3", "BMW X5", "BMW X6", "BYD F0", "BYD F3R", "BYD L3", "BYD S6", "Baic E150", "Baojun 630", "Brilliance FRV", "Brilliance FSV", "Brilliance M2", "Buick Encore", "Buick Excelle", "Buick Excelle GT", "Buick Excelle XT", "Buick GL8", "Buick GL8 II", "Buick Lacrosse", "Buick Regal", "Buick Regal GS", "Cadillac CTS", "Cadillac SRX", "Cadillac Seville", "Cadillac XTS", "Changan Alsvin", "Changan Benni Mini", "Changan CX20", "Changan Eado", "Changan S35", "Chery A3", "Chery E5", "Chery Tiggo 3", "Chevrolet Aveo", "Chevrolet Camaro", "Chevrolet Captiva", "Chevrolet Cruze", "Chevrolet Epica", "Chevrolet Lova", "Chevrolet Malibu", "Chevrolet Sail", "Chevrolet Spark", "Chrysler 300C", "Citroen C-Elysee", "Citroen C-Quatre Hatchback", "Citroen C-Quatre Sedan", "Citroen C4L", "Citroen C5", "Citroen DS3", "Citroen DS5", "Dongfeng H30", "Dongfeng Joyear", "Dongfeng S30", "Dongfeng Succe", "Emgrand EC7-RV", "Emgrand EC715", "FAW Bestune B50", "FAW Besturn B70", "FAW Besturn B90", "FAW N5", "Fiat 500", "Fiat Bravo", "Ford Ecosport", "Ford Fiesta Hatchback", "Ford Fiesta Sedan", "Ford Focus Hatchback", "Ford Focus Sedan", "Ford Mondeo", "Ford Mustang", "GAC Trumpchi GS5", "Geely", "Great Wall Haval M4", "Great Wall Hover H3", "Great Wall Voleex C50", "Great Wall Voleex V80", "Haima 2", "Haima 3", "Haima 7", "Haima Freema", "Honda Accord", "Honda CR-V", "Honda Civic", "Honda Crosstour", "Hyundai Santa Fe", "Infinity Q70L", "Infinity QX50", "Infinity QX70", "JAC A13", "JAC B15", "JAC B18", "Jaguar XF", "Jaguar XJL", "Jeep Compass", "Jeep Patriot", "Jeep Wrangler", "Kia Cadenza", "Kia Forte", "Kia K5", "Kia Koup", "Kia Sorento", "Kia Soul", "Kia Sportage", "Kia Sportage R", "Lamborghini Gallardo", "Land Rover Discovery", "Land Rover Range Rover Evoque", "Land Rover Range Rover Sport", "Land Rover Range Rover Vogue", "Landwind X8", "Lexus CT200h", "Lexus GS", "Lexus GS h", "Lexus IS", "Lexus RX", "Lifan 320", "MG3", "MG6", "Mazda 2", "Mazda 3 Hatchback", "Mazda 3 Sedan", "Mazda 5", "Mazda 6 Atenza", "Mazda 6 GG1", "Mazda 6 GH", "Mazda CX-5", "Mercedes-AMG C63", "Mercedes-Benz C-Class Hatchback", "Mercedes-Benz C-Class Sedan", "Mercedes-Benz E-Class Convertible", "Mercedes-Benz E-Class Coupe", "Mercedes-Benz E-Class Sedan", "Mercedes-Benz GLK-Class", "Mercedes-Benz R-Class", "Mercedes-Benz S-Class", "Mercedes-Benz SLK-Class", "Mini Cooper", "Mini Cooper Clubman", "Mini Cooper Countryman", "Mini Cooper Paceman", "Mitsubishi Lancer", "Mitsubishi Lancer EX", "Mitsubishi Outlander", "Mitsubishi Pajero", "Nissan Bluebird Sylphy", "Nissan GT-R", "Nissan NV200", "Nissan Qashqai", "Nissan Sunny", "Nissan Teana", "Opel Antara", "Opel Astra GTC", "Peugeot 207 Hatchback", "Peugeot 207 Sedan", "Peugeot 307", "Peugeot 308", "Peugeot 408", "Peugeot 508", "Peugeot RCZ", "Porsche 911", "Porsche Cayenne", "Porsche Cayman", "Porsche Panamera", "Renault Koleos", "Riich G5", "Roewe 350", "Roewe 550", "Skoda Fabia", "Skoda Octavia", "Skoda Rapid", "Skoda Superb", "Subaru Impreza", "Suzuki Alto", "Suzuki Kizashi", "Suzuki SX4", "Suzuki Swift", "Suzuki X5", "Toyota Camry", "Toyota Corolla", "Toyota Crown", "Toyota EZ", "Toyota GT86", "Toyota Land Cruiser Prado", "Toyota RAV4", "Toyota Reiz", "Toyota Vios", "Toyota Yaris", "Volkswagen Bora", "Volkswagen CC", "Volkswagen Dune", "Volkswagen Eos", "Volkswagen GTI", "Volkswagen Golf", "Volkswagen Jetta", "Volkswagen Lavida", "Volkswagen Magotan", "Volkswagen Multivan", "Volkswagen Passat", "Volkswagen Phaeton", "Volkswagen Polo", "Volkswagen Sagitar", "Volkswagen Scirocco", "Volkswagen Tiguan", "Volkswagen Touareg", "Volkswagen Touran", "Volvo C30", "Volvo C70", "Volvo S40", "Volvo S60", "Volvo S60L", "Volvo S80L", "Volvo V40", "Volvo V60", "Volvo XC60", "Volvo XC90", "Wuling Hongguang"};

            // End timing
            long endTime = System.currentTimeMillis();
            long executionTime = endTime - startTime;

            modelTextView.setText(String.format("Model ales: EfficientNet"));
            modelTextView.setTextSize(30); // Increase text size
            modelTextView.setTypeface(null, Typeface.BOLD); // Set text to bold

            classTextView.setText(String.format("Clasa prezisă: %s", classes[maxPos]));
            classTextView.setTextSize(24); // Increase text size
            classTextView.setTypeface(null, Typeface.ITALIC); // Set text to bold and italic

            confidenceTextView.setText(String.format("Precizie: %.2f%%", maxConfidence * 100));
            confidenceTextView.setTextSize(24); // Increase text size
            confidenceTextView.setTypeface(null, Typeface.ITALIC); // Set text to italic

            executionTimeTextView.setText(String.format("Latență: %d ms", executionTime));
            executionTimeTextView.setTextSize(24); // Increase text size
            executionTimeTextView.setTypeface(null, Typeface.ITALIC); // Set text to bold

            // Make the TextViews visible
            modelTextView.setVisibility(View.VISIBLE);
            classTextView.setVisibility(View.VISIBLE);
            confidenceTextView.setVisibility(View.VISIBLE);
            executionTimeTextView.setVisibility(View.VISIBLE);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                imageView.setImageBitmap(image);
                classifyImage(image);
            } else if (requestCode == 1) {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
