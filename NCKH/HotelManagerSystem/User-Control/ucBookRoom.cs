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
using HotelManagerSystem.DAO;
using HotelManagerSystem.DTO;
using HotelManagerSystem.BUL;
using System.Media;
using MongoDB.Driver;
namespace HotelManagerSystem.User_Control
{
    public partial class ucBookRoom : UserControl
    {
        List<Object_DTO> listObjectDTO;
        SoundPlayer soundPlayer = new SoundPlayer();
        MongoClientSettings settings = new MongoClientSettings();
        public ucBookRoom()
        {
            InitializeComponent();
            dataGridViewCustomer.ColumnHeadersDefaultCellStyle.Font = new System.Drawing.Font("Segoe UI", 9.75F);
            //DataTable dt = GetFullObject();
            //Object_DTO obj_DTO;
            //if (dt != null && dt.Rows.Count > 0)
            //{
            //    listObjectDTO = new List<Object_DTO>();
            //    for (int i = 0; i < dt.Rows.Count; i++)
            //    {
            //        obj_DTO = new Object_DTO();
            //        obj_DTO.MaDT = dt.Rows[i]["MaDT"].ToString().Trim();
            //        obj_DTO.TenDT = dt.Rows[i]["TenDT"].ToString().Trim();
            //        obj_DTO.NgaySinh = dt.Rows[i]["NgaySinh"].ToString().Trim();
            //        obj_DTO.GioiTinh = dt.Rows[i]["GioiTinh"].ToString().Trim();
            //        obj_DTO.pathFile = dt.Rows[i]["pathFile"].ToString().Trim();
            //        obj_DTO.CMND = dt.Rows[i]["CMND"].ToString().Trim();
            //        obj_DTO.QuocTich = dt.Rows[i]["QuocTich"].ToString().Trim();
            //        obj_DTO.QueQuan = dt.Rows[i]["QueQuan"].ToString().Trim();
            //        obj_DTO.Phone = dt.Rows[i]["Phone"].ToString().Trim();

            //        listObjectDTO.Add(obj_DTO);
            //    }
            //}
            //LoadFullObject(GetFullObject());
            //comboboxID.DisplayMember = "MaDT";
            LoadData();
            display_1.Visible = true;
            stop.Visible = false;
        }
        private void LoadData()
        {
            settings.Server = new MongoServerAddress("localhost", 27017);
            MongoClient client = new MongoClient();
            var db = client.GetDatabase("local");
            var collection = db.GetCollection<Object_DTO>("Infor_Speaker");
            var query = collection.AsQueryable<Object_DTO>().ToList();

            dataGridViewCustomer.DataSource = query;

        }
        private void LoadFullObject(DataTable table)
        {
            BindingSource source = new BindingSource();
            source.DataSource = table;
            dataGridViewCustomer.DataSource = source;
            bindingCustomer.BindingSource = source;
            comboboxID.DataSource = source;
        }
        private DataTable GetFullObject()
        {
            return Object_BUL.LoadObject();
        }
        string pathFile = "";
        private void ChangeText(DataGridViewRow row)
        {
            if (row.IsNewRow)
            {
                txtFullName.Text = string.Empty;
                txtCMND.Text = string.Empty;
                txtPhone.Text = string.Empty;
                txbAddress.Text = string.Empty;
                txbQuocTich.Text = string.Empty;
                bindingNavigatorMoveFirstItem.Enabled = false;
                bindingNavigatorMovePreviousItem.Enabled = false;
            }
            else
            {
                bindingNavigatorMoveFirstItem.Enabled = true;
                bindingNavigatorMovePreviousItem.Enabled = true;
                //comboboxID.Text = row.Cells["colID"].Value.ToString();
                txtFullName.Text = row.Cells["colNameCustomer"].Value.ToString();
                txbAddress.Text = row.Cells["colAddress"].Value.ToString();
                txtCMND.Text = row.Cells["colIDCard"].Value.ToString();
                txbQuocTich.Text = row.Cells["colNationality"].Value.ToString();
                txtPhone.Text = row.Cells["colPhone"].Value.ToString();
                comboBoxSex.SelectedItem = row.Cells["colSex"].Value;
                txtFileName.Text = row.Cells["colpathFile"].Value.ToString();
                pathFile = txtFileName.Text;
                string a = row.Cells["colDateOfBirth"].Value.ToString().Trim();
                datepickerDateOfBirth.Value = DateTime.Parse(a);
                //Object_DTO object_DTO = new Object_DTO(((DataRowView)row.DataBoundItem).Row);
                //groupCustomer.Tag = object_DTO;
            }
        }
        private static bool CheckInFillText(Control[] controls)
        {
            foreach (var control in controls)
            {
                if (control.Text == string.Empty)
                    return false;
            }
            return true;
        }





        private void btnClose__Click(object sender, EventArgs e)
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
        private Object_DTO GetObjectNow()
        {
            Object_DTO object_DTO = new Object_DTO();
            object_DTO.TenDT = txtFullName.Text;
            object_DTO.Phone = txtPhone.Text;
            object_DTO.CMND = txtCMND.Text;
            object_DTO.GioiTinh = comboBoxSex.Text;
            object_DTO.QueQuan = txbAddress.Text;
            object_DTO.QuocTich = txbQuocTich.Text;
            object_DTO.NgaySinh = datepickerDateOfBirth.Value.ToShortDateString();
            object_DTO.PathFile = txtFileName.Text;

            return object_DTO;
        }
        private void updateObject()
        {
            if (comboboxID.Text == string.Empty)
            {
                MessageBox.Show("Đối tượng này chưa tồn tại", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Stop);
            }
            else
            {
                if (!CheckInFillText(new Control[] { txtFullName, txtCMND, txtPhone, txbAddress, txbQuocTich }))
                {
                    MessageBox.Show("Thông tin không được để trống", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                else
                {
                    Object_DTO objectPre = groupCustomer.Tag as Object_DTO;
                    try
                    {
                        Object_DTO objectDTONow = GetObjectNow();
                        if (objectDTONow.Equals(objectPre))
                        {
                            MessageBox.Show("Bạn chưa thay đổi dữ liệu", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                        }
                        else
                        {
                            if (Object_BUL.UpdateObject(objectDTONow))
                            {
                                MessageBox.Show("Cập nhật thành công!", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Information);
                                groupCustomer.Tag = objectDTONow;
                                int index = dataGridViewCustomer.SelectedRows[0].Index;
                                LoadFullObject(GetFullObject());
                                comboboxID.SelectedIndex = index;
                            }
                            else
                            {
                                MessageBox.Show("Cập nhật không thành công", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                            }
                        }
                    }
                    catch
                    {
                        MessageBox.Show("Cập nhật không thành công", "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                    }
                }
            }
        }
        private void btnUpdate_Click(object sender, EventArgs e)
        {
            //DialogResult result = MessageBox.Show("Bạn có muốn cập nhật đối tượng này không?", "Thông báo", MessageBoxButtons.OKCancel, MessageBoxIcon.Question, MessageBoxDefaultButton.Button1);
            //if (result == DialogResult.OK)
            //{
            //    updateObject();
            //}
        }

        private void btnRemove_Click(object sender, EventArgs e)
        {
            //DialogResult result = MessageBox.Show("Bạn có muốn xóa đối tượng này không", "Thông báo", MessageBoxButtons.OKCancel, MessageBoxIcon.Question, MessageBoxDefaultButton.Button1);
            //if (result == DialogResult.OK)
            //{

            //    if (Object_BUL.RemoveObject(comboboxID.Text))
            //    {
            //        MessageBox.Show("Xóa thành công", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Information);
            //        LoadFullObject(GetFullObject());
            //    }
            //    else
            //    {
            //        MessageBox.Show("Xóa thất bại", "Thông báo", MessageBoxButtons.OK, MessageBoxIcon.Information);
            //    }

            //}
        }

        private void btnThoatObj_Click(object sender, EventArgs e)
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

        private void btnAddObject_Click(object sender, EventArgs e)
        {
            new fAddObject().ShowDialog();           
            LoadData();
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






