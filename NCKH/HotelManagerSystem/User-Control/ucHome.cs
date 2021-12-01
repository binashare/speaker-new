using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using HotelManagerSystem;
using MetroFramework;
using MetroFramework.Controls;
using HotelManagerSystem.DAO;
namespace HotelManagerSystem.User_Control
{
    public partial class ucHome : UserControl
    {
        int changePointX = 0;
        string ucName = "";
        string userName;
        public ucHome()
        {
            InitializeComponent();
            
        }
       
      
        private void metro_Click_1(object sender, EventArgs e)
        {
            MetroTile btn = (MetroTile)sender;
            ucName = btn.Tag.ToString();           
            timerHome.Start();
        }

        private void timerHome_Tick_1(object sender, EventArgs e)
        {
            if (changePointX >= metroPanel2.Size.Width)
            {
                timerHome.Stop();
                switch (ucName)
                {
                    //case "ucManageBill":
                    //    {
                            
                    //            ucManageBill ucmanageBill = new ucManageBill();
                    //            ucmanageBill.Dock = DockStyle.Fill;
                    //            fHome.frmHome.MetroContainer.Controls.Add(ucmanageBill);
                    //            fHome.frmHome.MetroContainer.Controls["ucManageBill"].BringToFront();
                    //            //Duyet de Xoa cac phan tu trong MetroControl
                    //            foreach (ucHome uc in fHome.frmHome.MetroContainer.Controls.OfType<ucHome>())
                    //            {
                    //                fHome.frmHome.MetroContainer.Controls.Remove(uc);
                    //            }
                           
                    //    }; break;
                    case "ucManageRoom":
                        {
                           
                                ucManageRoom ucmanageRoom = new ucManageRoom();
                                ucmanageRoom.Dock = DockStyle.Fill;
                                fHome.frmHome.MetroContainer.Controls.Add(ucmanageRoom);
                                fHome.frmHome.MetroContainer.Controls["ucManageRoom"].BringToFront();
                                //Duyet de Xoa cac phan tu trong MetroControl
                                foreach (ucHome uc in fHome.frmHome.MetroContainer.Controls.OfType<ucHome>())
                                {
                                    fHome.frmHome.MetroContainer.Controls.Remove(uc);
                                }
                            
                        }; break;
                    case "ucServiceAndPay":
                        {
                            
                                ucServiceAndPay ucserviceandpay = new ucServiceAndPay();
                                ucserviceandpay.Dock = DockStyle.Fill;
                                fHome.frmHome.MetroContainer.Controls.Add(ucserviceandpay);
                                fHome.frmHome.MetroContainer.Controls["ucServiceAndPay"].BringToFront();
                                //Duyet de Xoa cac phan tu trong MetroControl
                                foreach (ucHome uc in fHome.frmHome.MetroContainer.Controls.OfType<ucHome>())
                                {
                                    fHome.frmHome.MetroContainer.Controls.Remove(uc);
                                }
                           
                        }; break;
                    case "ucBookRoom":
                        {
                            
                                ucBookRoom ucbookroom = new ucBookRoom();
                                ucbookroom.Dock = DockStyle.Fill;
                                fHome.frmHome.MetroContainer.Controls.Add(ucbookroom);
                                fHome.frmHome.MetroContainer.Controls["ucBookRoom"].BringToFront();
                                //Duyet de Xoa cac phan tu trong MetroControl
                                foreach (ucHome uc in fHome.frmHome.MetroContainer.Controls.OfType<ucHome>())
                                {
                                    fHome.frmHome.MetroContainer.Controls.Remove(uc);
                                }
                           
                        }; break;
                }
                
            }
            else
            {
                changePointX = changePointX + 50;
                metroPanel2.Location = new Point(changePointX, 14);
            }
        }

        
    }
}
