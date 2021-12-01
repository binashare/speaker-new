using Newtonsoft.Json.Linq;
using RestSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using HotelManagerSystem.DTO;
using HotelManagerSystem.BUL;
using MongoDB.Driver;
using MongoDB.Bson;

namespace HotelManagerSystem
{
    public partial class fAddObject : Form
    {
        public fAddObject()
        {
            InitializeComponent();
        }

        private void btnCloseAdd_Click(object sender, EventArgs e)
        {
            this.Close();
        }
        public static bool callApiPost(string path_1)
        {
            var client = new RestClient("http://127.0.0.1:8000/api/recognition/check_audio");
            client.Timeout = -1;
            var request = new RestRequest(Method.POST);

            request.AddFile("audio", path_1);
            request.AddParameter("multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW", "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample_101.pdf\"\r\nContent-Type: application/pdf\r\n\r\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--", ParameterType.RequestBody);
            IRestResponse response = client.Execute(request);
            var jObject = JObject.Parse(response.Content);

            bool result = (bool)jObject.GetValue("result");
            return result;

        }
        private void btnBrowse_2_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog1 = new OpenFileDialog
            {
                InitialDirectory = @"C:\",
                Title = "Browse WAV Files",

                CheckFileExists = true,
                CheckPathExists = true,

                DefaultExt = "wav",
                Filter = "wav files (*.wav)|*.wav",
                FilterIndex = 2,
                RestoreDirectory = true,
                ReadOnlyChecked = true,
                ShowReadOnly = true
            };

            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                if (callApiPost(openFileDialog1.FileName))
                    txtFileName_2.Text = openFileDialog1.FileName;
                else
                {
                    MessageBox.Show("File không hop lệ!", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }

            }
        }
        public static bool CheckFillInText(Control[] controls)
        {
            foreach (var control in controls)
            {
                if (control.Text.ToString() == string.Empty)
                {
                    return false;
                }
            }
            return true;

        }
        private Object_DTO GetObjectNow()
        {
            Object_DTO object_DTO = new Object_DTO();
           
            object_DTO.TenDT = txtFullName_2.Text;
            object_DTO.GioiTinh = comboBoxSex_2.Text;
            object_DTO.QuocTich = txbNationality_2.Text;
            object_DTO.QueQuan = txbAddress_2.Text;
            object_DTO.PathFile = txtFileName_2.Text;
            object_DTO.NgaySinh = datepickerDateOfBirth.Value.ToShortDateString();
            object_DTO.CMND = txtCMND_2.Text;
            object_DTO.Phone = txtNumberPhone_2.Text;
            return object_DTO;
        }
        private void Clean()
        {
            txtFullName_2.Text = string.Empty;
            txtCMND_2.Text = string.Empty;
            txtNumberPhone_2.Text = string.Empty;
            txtFileName_2.Text = string.Empty;
        }
        private void InsertObject()
        {
            if (!CheckFillInText(new Control[] { txtFileName_2, txtFullName_2, txtCMND_2, txtNumberPhone_2, txbAddress_2, txbNationality_2, comboBoxSex_2 }))
            {
                MessageBox.Show("Không được để trống", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            try
            {
                Object_DTO object_DTO = GetObjectNow();
                const string connectionString = "mongodb://localhost:27017";
                var client = new MongoClient(connectionString);
                var Client = new MongoClient();
                var MongoDB = Client.GetDatabase("local");
                var Collec = MongoDB.GetCollection<BsonDocument>("Infor_Speaker");
                var documnt = new BsonDocument
                {
                    {"CMND",object_DTO.CMND},
                    {"TenDT",object_DTO.TenDT},
                    {"GioiTinh",object_DTO.GioiTinh},
                    {"QuocTich",object_DTO.QuocTich},
                    {"QueQuan",object_DTO.QueQuan},
                    {"NgaySinh",object_DTO.NgaySinh},
                    {"Phone",object_DTO.Phone},
                    {"PathFile",object_DTO.PathFile },
                    {"Feature", BsonArray.Create(new double[]{ })}
                 };
                var filter = Builders<BsonDocument>.Filter.Eq("CMND", object_DTO.CMND);
                var result = Collec.Find(filter).FirstOrDefault();
                if (result == null)
                {
                    Collec.InsertOneAsync(documnt);
                    if (callApiUpdateFeature(object_DTO.PathFile, object_DTO.CMND) == 200)
                    {
                        MessageBox.Show("Thêm thành công", "Thông Báo", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    Clean();
                }
                else
                {
                    MessageBox.Show("CMND đã tồn tại!", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                }
            }
            catch
            {
                MessageBox.Show("Lỗi thêm đối tượng", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        public static int callApiUpdateFeature(string path_1, string CMND)
        {
            var client = new RestClient("http://127.0.0.1:8000/api/user/add_Speaker");
            client.Timeout = -1;
            var request = new RestRequest(Method.GET);

            request.AddParameter("audio", path_1);
            request.AddParameter("CMND", CMND);
            request.AddHeader("content-type", "multipart/form-data; boundary=-----------------------------28947758029299");
            IRestResponse response = client.Execute(request);
            var jObject = JObject.Parse(response.Content);

            int result = (int)jObject.GetValue("code");
            return result;

        }
        private void btnAddObject_2_Click(object sender, EventArgs e)
        {
            DialogResult result = MessageBox.Show("Bạn có muốn thêm đối tượng mới?", "Thông báo", MessageBoxButtons.OKCancel, MessageBoxIcon.Question, MessageBoxDefaultButton.Button1);
            if (result == DialogResult.OK)
            {
                InsertObject();
            }
        }
    }
}
