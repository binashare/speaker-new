using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using HotelManagerSystem.DAO;
using HotelManagerSystem.DTO;
using HotelManagerSystem.BUL;
using RestSharp;

namespace HotelManagerSystem
{
    public partial class fLogin : Form
    {
        List<Object_DTO> listObjectDTO;
        List<Feature_DTO> listFeatureDTO = new List<Feature_DTO>();
        public fLogin()
        {
            InitializeComponent();
            //DataTable dt = GetFullObject();
            //Object_DTO obj_DTO;
            //if (dt != null && dt.Rows.Count > 0)
            //{
            //    listObjectDTO = new List<Object_DTO>();
            //    for (int i = 0; i < dt.Rows.Count; i++)
            //    {
            //        obj_DTO = new Object_DTO();

            //        obj_DTO.TenDT = dt.Rows[i]["TenDT"].ToString().Trim();
            //        obj_DTO.NgaySinh = dt.Rows[i]["NgaySinh"].ToString().Trim();
            //        obj_DTO.GioiTinh = dt.Rows[i]["GioiTinh"].ToString().Trim();
            //        obj_DTO.pathFile = dt.Rows[i]["pathFile"].ToString().Trim();
            //        obj_DTO.CMND = dt.Rows[i]["CMND"].ToString().Trim();
            //        obj_DTO.QuocTich = dt.Rows[i]["QuocTich"].ToString().Trim();
            //        obj_DTO.QueQuan = dt.Rows[i]["QueQuan"].ToString().Trim();
            //        obj_DTO.Phone = dt.Rows[i]["Phone"].ToString().Trim();
            //        obj_DTO.pathFile = dt.Rows[i]["pathFile"].ToString().Trim();
            //        listObjectDTO.Add(obj_DTO);
            //    }
            //}
        }
       
      


        private void btnExit__Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void btnDangnhap_Click(object sender, EventArgs e)
        {
            string username = txtUsername.Text;
            string matkhau = txtMK.Text;

            if (username == "admin" && matkhau == "admin")
            {
              //  getFeature();
                fHome f = new fHome();
                this.Hide();
                f.ShowDialog();

            }
            else
            {
                MessageBox.Show("Mật khẩu hoặc tên đăng nhập không chính xác!\nVui lòng nhập lại", "Lỗi!", MessageBoxButtons.OK, MessageBoxIcon.Error);

            }
        }
        //private void getFeature()
        //{
        //    for (int i = 0; i < listObjectDTO.Count; i++)
        //    {
        //        Feature_DTO featureDTO = new Feature_DTO();
        //        callApiGetFeature(listObjectDTO[i].pathFile);
        //        Console.WriteLine(featureDTO.feature);
        //        listFeatureDTO.Add(featureDTO);
        //    }
        //}
        private void bunifuImageButton1_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }
        //private DataTable GetFullObject()
        //{
        //    return Object_BUL.LoadObject();
        //}
        //private static void callApiGetFeature(string path_1)
        //{
        //    var client = new RestClient("http://127.0.0.1:8000/api/recognition/get_feature");
        //    client.Timeout = -1;
        //    var request = new RestRequest(Method.GET);
        //    request.AddParameter("audio", path_1);
        //    IRestResponse response = client.Execute(request);

        //}

        private void btnThoat_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void label6_Click(object sender, EventArgs e)
        {

        }

        private void label7_Click(object sender, EventArgs e)
        {

        }
    }
}
