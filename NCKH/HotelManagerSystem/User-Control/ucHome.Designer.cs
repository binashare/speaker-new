
namespace HotelManagerSystem.User_Control
{
    partial class ucHome
    {
        /// <summary> 
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ucHome));
            this.metroPanel1 = new MetroFramework.Controls.MetroPanel();
            this.metroPanel2 = new MetroFramework.Controls.MetroPanel();
            this.metroTile16 = new MetroFramework.Controls.MetroTile();
            this.titleManageRoom = new MetroFramework.Controls.MetroTile();
            this.titlePay = new MetroFramework.Controls.MetroTile();
            this.titleBookRoom = new MetroFramework.Controls.MetroTile();
            this.timerHome = new System.Windows.Forms.Timer(this.components);
            this.metroPanel1.SuspendLayout();
            this.metroPanel2.SuspendLayout();
            this.SuspendLayout();
            // 
            // metroPanel1
            // 
            this.metroPanel1.Controls.Add(this.metroPanel2);
            this.metroPanel1.HorizontalScrollbarBarColor = true;
            this.metroPanel1.HorizontalScrollbarHighlightOnWheel = false;
            this.metroPanel1.HorizontalScrollbarSize = 10;
            this.metroPanel1.Location = new System.Drawing.Point(3, 3);
            this.metroPanel1.Name = "metroPanel1";
            this.metroPanel1.Size = new System.Drawing.Size(1126, 737);
            this.metroPanel1.TabIndex = 0;
            this.metroPanel1.VerticalScrollbarBarColor = true;
            this.metroPanel1.VerticalScrollbarHighlightOnWheel = false;
            this.metroPanel1.VerticalScrollbarSize = 10;
            this.metroPanel1.Click += new System.EventHandler(this.metro_Click_1);
            // 
            // metroPanel2
            // 
            this.metroPanel2.Controls.Add(this.metroTile16);
            this.metroPanel2.Controls.Add(this.titleManageRoom);
            this.metroPanel2.Controls.Add(this.titlePay);
            this.metroPanel2.Controls.Add(this.titleBookRoom);
            this.metroPanel2.HorizontalScrollbarBarColor = true;
            this.metroPanel2.HorizontalScrollbarHighlightOnWheel = false;
            this.metroPanel2.HorizontalScrollbarSize = 10;
            this.metroPanel2.Location = new System.Drawing.Point(0, 0);
            this.metroPanel2.Name = "metroPanel2";
            this.metroPanel2.Size = new System.Drawing.Size(1123, 734);
            this.metroPanel2.TabIndex = 2;
            this.metroPanel2.VerticalScrollbarBarColor = true;
            this.metroPanel2.VerticalScrollbarHighlightOnWheel = false;
            this.metroPanel2.VerticalScrollbarSize = 10;
            // 
            // metroTile16
            // 
            this.metroTile16.ActiveControl = null;
            this.metroTile16.BackColor = System.Drawing.Color.SeaGreen;
            this.metroTile16.ForeColor = System.Drawing.Color.White;
            this.metroTile16.Location = new System.Drawing.Point(747, 43);
            this.metroTile16.Margin = new System.Windows.Forms.Padding(4);
            this.metroTile16.Name = "metroTile16";
            this.metroTile16.Size = new System.Drawing.Size(218, 381);
            this.metroTile16.TabIndex = 86;
            this.metroTile16.Tag = "ucManageBill";
            this.metroTile16.Text = "Quy định";
            this.metroTile16.TileImage = ((System.Drawing.Image)(resources.GetObject("metroTile16.TileImage")));
            this.metroTile16.TileImageAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.metroTile16.TileTextFontWeight = MetroFramework.MetroTileTextWeight.Regular;
            this.metroTile16.UseCustomBackColor = true;
            this.metroTile16.UseCustomForeColor = true;
            this.metroTile16.UseSelectable = true;
            this.metroTile16.UseStyleColors = true;
            this.metroTile16.UseTileImage = true;
            // 
            // titleManageRoom
            // 
            this.titleManageRoom.ActiveControl = null;
            this.titleManageRoom.BackColor = System.Drawing.Color.SeaGreen;
            this.titleManageRoom.ForeColor = System.Drawing.Color.White;
            this.titleManageRoom.Location = new System.Drawing.Point(462, 43);
            this.titleManageRoom.Margin = new System.Windows.Forms.Padding(4);
            this.titleManageRoom.Name = "titleManageRoom";
            this.titleManageRoom.Size = new System.Drawing.Size(219, 381);
            this.titleManageRoom.TabIndex = 83;
            this.titleManageRoom.Tag = "ucManageRoom";
            this.titleManageRoom.Text = "Real Time";
            this.titleManageRoom.TileImage = ((System.Drawing.Image)(resources.GetObject("titleManageRoom.TileImage")));
            this.titleManageRoom.TileImageAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.titleManageRoom.TileTextFontWeight = MetroFramework.MetroTileTextWeight.Regular;
            this.titleManageRoom.UseCustomBackColor = true;
            this.titleManageRoom.UseCustomForeColor = true;
            this.titleManageRoom.UseSelectable = true;
            this.titleManageRoom.UseStyleColors = true;
            this.titleManageRoom.UseTileImage = true;
            this.titleManageRoom.Click += new System.EventHandler(this.metro_Click_1);
            // 
            // titlePay
            // 
            this.titlePay.ActiveControl = null;
            this.titlePay.BackColor = System.Drawing.Color.SeaGreen;
            this.titlePay.ForeColor = System.Drawing.Color.White;
            this.titlePay.Location = new System.Drawing.Point(179, 458);
            this.titlePay.Margin = new System.Windows.Forms.Padding(4);
            this.titlePay.Name = "titlePay";
            this.titlePay.Size = new System.Drawing.Size(786, 190);
            this.titlePay.TabIndex = 80;
            this.titlePay.Tag = "ucServiceAndPay";
            this.titlePay.Text = "Truy tìm đối tượng";
            this.titlePay.TileImage = ((System.Drawing.Image)(resources.GetObject("titlePay.TileImage")));
            this.titlePay.TileImageAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.titlePay.TileTextFontWeight = MetroFramework.MetroTileTextWeight.Regular;
            this.titlePay.UseCustomBackColor = true;
            this.titlePay.UseCustomForeColor = true;
            this.titlePay.UseSelectable = true;
            this.titlePay.UseStyleColors = true;
            this.titlePay.UseTileImage = true;
            this.titlePay.Click += new System.EventHandler(this.metro_Click_1);
            // 
            // titleBookRoom
            // 
            this.titleBookRoom.ActiveControl = null;
            this.titleBookRoom.BackColor = System.Drawing.Color.SeaGreen;
            this.titleBookRoom.ForeColor = System.Drawing.Color.White;
            this.titleBookRoom.Location = new System.Drawing.Point(179, 43);
            this.titleBookRoom.Margin = new System.Windows.Forms.Padding(4);
            this.titleBookRoom.Name = "titleBookRoom";
            this.titleBookRoom.Size = new System.Drawing.Size(206, 381);
            this.titleBookRoom.TabIndex = 78;
            this.titleBookRoom.Tag = "ucBookRoom";
            this.titleBookRoom.Text = "Danh sách đối tượng";
            this.titleBookRoom.TileImage = ((System.Drawing.Image)(resources.GetObject("titleBookRoom.TileImage")));
            this.titleBookRoom.TileImageAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.titleBookRoom.TileTextFontWeight = MetroFramework.MetroTileTextWeight.Regular;
            this.titleBookRoom.UseCustomBackColor = true;
            this.titleBookRoom.UseCustomForeColor = true;
            this.titleBookRoom.UseSelectable = true;
            this.titleBookRoom.UseStyleColors = true;
            this.titleBookRoom.UseTileImage = true;
            this.titleBookRoom.Click += new System.EventHandler(this.metro_Click_1);
            // 
            // timerHome
            // 
            this.timerHome.Interval = 10;
            this.timerHome.Tick += new System.EventHandler(this.timerHome_Tick_1);
            // 
            // ucHome
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.Controls.Add(this.metroPanel1);
            this.Name = "ucHome";
            this.Size = new System.Drawing.Size(1110, 730);
            this.metroPanel1.ResumeLayout(false);
            this.metroPanel2.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private MetroFramework.Controls.MetroPanel metroPanel1;
        private System.Windows.Forms.Timer timerHome;
        public  MetroFramework.Controls.MetroPanel metroPanel2;
        private MetroFramework.Controls.MetroTile metroTile16;
        private MetroFramework.Controls.MetroTile titleManageRoom;
        private MetroFramework.Controls.MetroTile titlePay;
        private MetroFramework.Controls.MetroTile titleBookRoom;
    }
}
