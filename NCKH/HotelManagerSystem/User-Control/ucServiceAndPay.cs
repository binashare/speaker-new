using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Media;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using HotelManagerSystem.BUL;
using HotelManagerSystem.DAO;
using HotelManagerSystem.DTO;
using Newtonsoft.Json.Linq;
using RestSharp;
using MongoDB.Driver;

namespace HotelManagerSystem.User_Control
{
    public partial class ucServiceAndPay : UserControl
    {
        List<Object_DTO> listObjectDTO;
        List<Object_DTO> listResult = new List<Object_DTO>();
        public SoundPlayer soundPlayer;
        string pathFile = "";
        string _pathFile = "";

        MongoClientSettings settings = new MongoClientSettings();
        public ucServiceAndPay()
        {
            InitializeComponent();
            DataTable dt = GetFullObject();
            Object_DTO obj_DTO;
            display_1.Visible = true;
            stop.Visible = false;
            display_2.Visible = true;
            stop_2.Visible = false;
            if (dt != null && dt.Rows.Count > 0)
            {
                listObjectDTO = new List<Object_DTO>();
                for (int i = 0; i < dt.Rows.Count; i++)
                {
                    obj_DTO = new Object_DTO();

                    obj_DTO.TenDT = dt.Rows[i]["TenDT"].ToString().Trim();
                    obj_DTO.NgaySinh = dt.Rows[i]["NgaySinh"].ToString().Trim();
                    obj_DTO.GioiTinh = dt.Rows[i]["GioiTinh"].ToString().Trim();
                    obj_DTO.PathFile = dt.Rows[i]["pathFile"].ToString().Trim();
                    obj_DTO.CMND = dt.Rows[i]["CMND"].ToString().Trim();
                    obj_DTO.QuocTich = dt.Rows[i]["QuocTich"].ToString().Trim();
                    obj_DTO.QueQuan = dt.Rows[i]["QueQuan"].ToString().Trim();
                    obj_DTO.Phone = dt.Rows[i]["Phone"].ToString().Trim();

                    listObjectDTO.Add(obj_DTO);
                }
            }
        }
        private DataTable GetFullObject()
        {
            return Object_BUL.LoadObject();
        }
        private void btnClose__Click(object sender, EventArgs e)
        {
           
            ucHome uchome = new ucHome();
            uchome.Dock = DockStyle.Fill;
            fHome.frmHome.MetroContainer.Controls.Add(uchome);
            fHome.frmHome.MetroContainer.Controls["ucHome"].BringToFront();
            foreach (ucServiceAndPay uc in fHome.frmHome.MetroContainer.Controls.OfType<ucServiceAndPay>())
            {
                fHome.frmHome.MetroContainer.Controls.Remove(uc);
            }
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
        private void btnBrowse_Click(object sender, EventArgs e)
        {
            dataGridViewCustomer.DataSource = null;
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
                {
                    txtFileName.Text = openFileDialog1.FileName;
                    pathFile = txtFileName.Text;

                }
                else
                {
                    MessageBox.Show("Định dạng file âm thanh không hợp lệ!", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void display_1_Click(object sender, EventArgs e)
        {
            if (pathFile != "")
            {
                soundPlayer = new SoundPlayer(pathFile);
                soundPlayer.Play();
                stop.Visible = true;
                display_1.Visible = false;
            }
            else
            {
                MessageBox.Show("Bạn chưa chọn file!", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void stop_Click(object sender, EventArgs e)
        {
            soundPlayer.Stop();
            stop.Visible = false;
            display_1.Visible = true;
        }

        private void btnThoatRecog_Click(object sender, EventArgs e)
        {
            ucHome uchome = new ucHome();
            uchome.Dock = DockStyle.Fill;
            fHome.frmHome.MetroContainer.Controls.Add(uchome);
            fHome.frmHome.MetroContainer.Controls["ucHome"].BringToFront();
            foreach (ucServiceAndPay uc in fHome.frmHome.MetroContainer.Controls.OfType<ucServiceAndPay>())
            {
                fHome.frmHome.MetroContainer.Controls.Remove(uc);
            }
        }
        List<string> x = new List<string>();
        private void btnSearch_Click(object sender, EventArgs e)
        {
            dataGridViewCustomer.DataSource = null;
            dataGridViewCustomer.Refresh();
            x.Clear();
            listResult.Clear();
            if (pathFile == "")
            {
                MessageBox.Show("Bạn chưa chọn file", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else
            {
                if (callApiTop_5(pathFile) == true)
                {
                    string fileText = @"C:\DoAnToNghiep\Speaker_New\NCKH\text.txt";

                    var lines = File.ReadAllLines(fileText);
                    if (lines.Length == 0)
                    {
                        MessageBox.Show("không tìm thấy đối tượng khả nghi trong cơ sở dữ liệu", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        for (var i = 0; i < lines.Length; i += 1)
                        {

                            var line = lines[i];
                            string CMND = line.Split(':')[0].ToString();
                            x.Add(CMND);
                          
                        }               

                        settings.Server = new MongoServerAddress("localhost", 27017);
                        MongoClient client = new MongoClient();
                        var db = client.GetDatabase("local");
                        var collection = db.GetCollection<Object_DTO>("Infor_Speaker");

                        var query = collection.AsQueryable<Object_DTO>().Where(a => x.Contains(a.CMND)).ToList();
                        dataGridViewCustomer.DataSource = query;

                    }


                }
            }
        }
        private static bool callApiTop_5(string path_1)
        {
            var client = new RestClient("http://127.0.0.1:8000/api/recognition/get_Top5");
            client.Timeout = -1;
            var request = new RestRequest(Method.POST);
            request.AddFile("audio", path_1);
            IRestResponse response = client.Execute(request);
            var jObject = JObject.Parse(response.Content);
            bool result = (bool)jObject.GetValue("result");
            return result;
        }


        private void ChangeText(DataGridViewRow row)
        {
            if (row.IsNewRow)
            {
                txtFullName.Text = string.Empty;
                txtCMND.Text = string.Empty;
                txtPhone.Text = string.Empty;
                txbAddress.Text = string.Empty;
                txbNationality.Text = string.Empty;
            }
            else
            {
               
                txtFullName.Text = row.Cells[0].Value.ToString();
                datepickerDateOfBirth.Value = Convert.ToDateTime(row.Cells[1].Value.ToString());
                comboBoxSex.SelectedItem = row.Cells[2].Value;
                txtCMND.Text = row.Cells[4].Value.ToString();
                txbNationality.Text = row.Cells[6].Value.ToString();
                txbAddress.Text = row.Cells[7].Value.ToString();
                txtPhone.Text = row.Cells[5].Value.ToString();
                txtFileName_1.Text = row.Cells[3].Value.ToString();
                _pathFile = row.Cells[3].Value.ToString();
            }
        }
        private void display_2_Click(object sender, EventArgs e)
        {
            if (_pathFile != "")
            {
                soundPlayer = new SoundPlayer(_pathFile);
                soundPlayer.Play();
                stop_2.Visible = true;
                display_2.Visible = false;
            }
            else
            {
                MessageBox.Show("Bạn chưa chọn file!", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void stop_2_Click(object sender, EventArgs e)
        {
            soundPlayer.Stop();
            stop_2.Visible = false;
            display_2.Visible = true;
        }

        private void dataGridViewCustomer_SelectionChanged_1(object sender, EventArgs e)
        {
            if (dataGridViewCustomer.SelectedRows.Count > 0)
            {
                DataGridViewRow row = dataGridViewCustomer.SelectedRows[0];
                ChangeText(row);
            }
        }
    }
}
