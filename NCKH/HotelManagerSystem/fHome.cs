using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MetroFramework;
using MetroFramework.Controls;
using HotelManagerSystem.User_Control;
using HotelManagerSystem.DAO;
using RestSharp;
using Newtonsoft.Json.Linq;

namespace HotelManagerSystem
{
    public partial class fHome : Form
    {
       
        private static fHome _frmHome;
        public static fHome frmHome
        {
            get
            {
                if (_frmHome == null) _frmHome = new fHome();
                return _frmHome;
            }
        }
        public MetroPanel MetroContainer
        {
            get
            {
                return this.metroPanelHome;
            }
            set
            {
                this.metroPanelHome = value;
            }
        }
        public fHome()
        {
            InitializeComponent();
            
        }
        
        private void fHome_Load(object sender, EventArgs e)
        {
            _frmHome = this;
            //Khởi tạo ucHome
            ucHome uchome = new ucHome();
            uchome.Dock = DockStyle.Fill;
            _frmHome.MetroContainer.Controls.Add(uchome);
            _frmHome.MetroContainer.Controls["ucHome"].BringToFront();

           
        }

        private void btnHome_Click(object sender, EventArgs e)
        {
            ucHome uchome = new ucHome();
            uchome.Dock = DockStyle.Fill;
            fHome.frmHome.MetroContainer.Controls.Add(uchome);
            fHome.frmHome.MetroContainer.Controls["ucHome"].BringToFront();
            foreach (ucBookRoom uc in fHome.frmHome.MetroContainer.Controls.OfType<ucBookRoom>())
            {
                fHome.frmHome.MetroContainer.Controls.Remove(uc);
            }


        }

        private void btnUser_Click(object sender, EventArgs e)
        {
            
        }

        private void btnLogout_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Bạn có muốn đăng xuất không?", "Thông báo", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
            {
                if (callApiReset() == true)
                {
                    fLogin f = new fLogin();
                    Hide();
                    f.ShowDialog();
                }

            }
        }
        private static bool callApiReset()
        {
            var client = new RestClient("http://127.0.0.1:8000/api/recognition/reset");
            client.Timeout = -1;
            var request = new RestRequest(Method.GET);
            IRestResponse response = client.Execute(request);
            var jObject = JObject.Parse(response.Content);
            bool result = (bool)jObject.GetValue("result");
            return result;

        }
        private void gunaControlBox1_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Bạn có muốn đăng xuất không?", "Thông báo", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
            {
                if (callApiReset() == true)
                {
                    Application.Exit();
                }

            }
        }

        private void metroPanelHome_Paint(object sender, PaintEventArgs e)
        {

        }
    }
}
