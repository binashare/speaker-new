using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using HotelManagerSystem.BUL;
namespace HotelManagerSystem.User_Control
{
    public partial class ucManageBill : UserControl
    {
        public ucManageBill()
        {
            InitializeComponent();
            dataGridViewBill.Font = new Font("Segoe UI", 9.75F);
            cmbID.DisplayMember = "MAHOADON";
        }
        private void LoadFullBill(DataTable table)
        {
            BindingSource source = new BindingSource();
            ChangePrice(table);
            source.DataSource = table;
            dataGridViewBill.DataSource = source;
            bindingBill.BindingSource = source;
            cmbID.DataSource = source;

            txtDayCreate.DataBindings.Clear();
           // txtNameRoom.DataBindings.Clear();
            txtPrice.DataBindings.Clear();
            txtStatusRoom.DataBindings.Clear(); 
            txtUser.DataBindings.Clear();
            txtDiscount.DataBindings.Clear();
            txtFinalPrice.DataBindings.Clear();

            txtDayCreate.DataBindings.Add("Text", source, "NGAYLAP");
           // txtNameRoom.DataBindings.Add("Text", source, "TenPhong");
            txtPrice.DataBindings.Add("Text", source, "TongGia");
            txtStatusRoom.DataBindings.Add("Text", source, "TINHTRANG");
            txtUser.DataBindings.Add("Text", source, "MaNV");
            txtDiscount.DataBindings.Add("Text", source, "KhuyenMai");
            txtFinalPrice.DataBindings.Add("Text", source, "ThanhTien");

        }
        private void btnCloseBill_Click(object sender, EventArgs e)
        {
            ucHome uchome = new ucHome();
            uchome.Dock = DockStyle.Fill;
            fHome.frmHome.MetroContainer.Controls.Add(uchome);
            fHome.frmHome.MetroContainer.Controls["ucHome"].BringToFront();
            foreach (ucManageBill uc in fHome.frmHome.MetroContainer.Controls.OfType<ucManageBill>())
            {
                fHome.frmHome.MetroContainer.Controls.Remove(uc);
            }
        }
        #region GetData
      
        #endregion GetData

        #region Utils
        private void ChangePrice(DataTable table)
        {
           // table.Columns.Add("TongTien_New", typeof(string));
            table.Columns.Add("ThanhTien_New", typeof(string));
            table.Columns.Add("TinhTrang_New", typeof(string));
            for (int i = 0; i < table.Rows.Count; i++)
            {
                if (table.Rows[i]["ThanhTien"].ToString() != "")
                    table.Rows[i]["ThanhTien_New"] = int.Parse(table.Rows[i]["ThanhTien"].ToString()).ToString("C0", CultureInfo.CreateSpecificCulture("vi-VN"));
                // table.Rows[i]["TongTien_New"] = int.Parse(table.Rows[i]["TongGia"].ToString()).ToString("C0", CultureInfo.CreateSpecificCulture("vi-VN"));
                else
                    table.Rows[i]["ThanhTien_New"] = int.Parse("0").ToString("C0", CultureInfo.CreateSpecificCulture("vi-VN"));
                if((bool)table.Rows[i]["TINHTRANG"] == true)
                {
                    table.Rows[i]["TinhTrang_New"] = "Đã thanh toán";
                }
                else
                    table.Rows[i]["TinhTrang_New"] = "Chưa thanh toán";
            }
            table.Columns.Remove("TINHTRANG");
            table.Columns.Remove("ThanhTien");
            // table.Columns.Remove("TongGia");
            // table.Columns["TongTien_New"].ColumnName = "TongTien";
            table.Columns["TinhTrang_New"].ColumnName = "TinhTrang";
            table.Columns["ThanhTien_New"].ColumnName = "ThanhTien";
        }
        #endregion

        private void dataGridViewBill_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        
    }
}
