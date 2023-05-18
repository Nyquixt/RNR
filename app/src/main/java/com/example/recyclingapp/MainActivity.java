package com.example.recyclingapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.NameValuePair;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.params.BasicHttpParams;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;
import org.apache.http.util.EntityUtils;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;


public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CODE2 = 17;

    //server address
    //emulator:
    //private static final String SERVER_ADDRESS = "http://10.0.2.2:5000/predict";
    //physical phone: usb debugging run "adb reverse tcp:3333 tcp:5000"
    private static final String SERVER_ADDRESS = "http://127.0.0.1:3333/predict";

    ImageButton cameraButton;
    ImageView imageView;
    TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraButton = findViewById(R.id.imageButton);
        textView = findViewById(R.id.resultText);
        imageView = findViewById(R.id.imageview);


        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(camera,REQUEST_CODE2);

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode == REQUEST_CODE2 && resultCode == RESULT_OK)
        {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            textView.setText("");
            imageView.setImageBitmap(photo);

            new UploadImage(photo, "inputImage").execute();
        }
        else {
            Toast.makeText(this, "Cancelled", Toast.LENGTH_SHORT).show();
            super.onActivityResult(requestCode, resultCode, data);
        }
    }

    private class UploadImage extends AsyncTask<Void, Void, Void> {
        Bitmap image;
        String name;

        public UploadImage(Bitmap image, String name){
            this.image = image;
            this.name = name;
        }

        public String getRecyclingInfo(String trashClass){
            String recyclingInfo = "";
            switch(trashClass){
                case "cardboard":
                    recyclingInfo = "Recyclable: Fold and place in bin";
                    break;
                case "e-waste":
                    recyclingInfo = "Not Recyclable: Bring to e-waste disposal facility";
                    break;
                case "glass":
                case "metal":
                case "plastic":
                    recyclingInfo = "Recyclable: Clean, dry off and place in bin";
                    break;
                case "paper":
                    recyclingInfo = "Recyclable: Place in bin";
                case "medical":
                    recyclingInfo = "Not Recyclable: Dispose at hospital";
                    break;
                default:
                    break;
            }
            return recyclingInfo;
        }

        public void displayResponse(String response) {
            String []splitterString=response.split("\"");
            response = splitterString[3].trim();
            response = response + "\n" + getRecyclingInfo(response);

            textView = findViewById(R.id.resultText);
            textView.setText(response);
        }

        @Override
        protected Void doInBackground(Void... voids) {

            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
            String encodedImage = Base64.encodeToString(byteArrayOutputStream.toByteArray(), Base64.DEFAULT);

            ArrayList<NameValuePair> dataToSend = new ArrayList<>();
            dataToSend.add(new BasicNameValuePair("image", encodedImage));
            dataToSend.add(new BasicNameValuePair("name", name));

            HttpParams httpRequestParams = getHttpRequestParams();

            HttpClient client = new DefaultHttpClient(httpRequestParams);
            HttpPost post = new HttpPost(SERVER_ADDRESS);
            HttpResponse response;

            try {
                post.setEntity(new UrlEncodedFormEntity(dataToSend));
                response = client.execute(post);
                HttpEntity entity = response.getEntity();
                displayResponse(EntityUtils.toString(entity));

            } catch (Exception e) {
                e.printStackTrace();
            }

            return null;
        }

        @Override
        protected void onPostExecute(Void unused) {
            super.onPostExecute(unused);
            Toast.makeText(getApplicationContext(), "Image Uploaded", Toast.LENGTH_SHORT).show();

        }
    }

    private HttpParams getHttpRequestParams(){
        HttpParams httpRequestParams = new BasicHttpParams();
        HttpConnectionParams.setConnectionTimeout(httpRequestParams, 1000 * 300);
        HttpConnectionParams.setSoTimeout(httpRequestParams, 1000*300);
        return httpRequestParams;
    }
}