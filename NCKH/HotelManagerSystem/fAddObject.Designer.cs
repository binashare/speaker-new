
namespace HotelManagerSystem
{
    partial class fAddObject
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

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(fAddObject));
            this.btnCloseAdd = new Bunifu.UI.WinForms.BunifuImageButton();
            this.txbAddress_2 = new MetroFramework.Controls.MetroComboBox();
            this.txbNationality_2 = new MetroFramework.Controls.MetroComboBox();
            this.label14 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.comboBoxSex_2 = new MetroFramework.Controls.MetroComboBox();
            this.label13 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.txtFileName_2 = new Bunifu.Framework.UI.BunifuMetroTextbox();
            this.datepickerDateOfBirth = new Bunifu.Framework.UI.BunifuDatepicker();
            this.btnBrowse_2 = new Bunifu.Framework.UI.BunifuThinButton2();
            this.txtNumberPhone_2 = new Bunifu.Framework.UI.BunifuMetroTextbox();
            this.txtCMND_2 = new Bunifu.Framework.UI.BunifuMetroTextbox();
            this.txtFullName_2 = new Bunifu.Framework.UI.BunifuMetroTextbox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.groupCustomer = new System.Windows.Forms.GroupBox();
            this.label16 = new System.Windows.Forms.Label();
            this.bunifuSeparator1 = new Bunifu.Framework.UI.BunifuSeparator();
            this.label2 = new System.Windows.Forms.Label();
            this.btnClose_2 = new Bunifu.Framework.UI.BunifuThinButton2();
            this.btnAddObject_2 = new Bunifu.Framework.UI.BunifuThinButton2();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.groupBox2.SuspendLayout();
            this.groupCustomer.SuspendLayout();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnCloseAdd
            // 
            this.btnCloseAdd.ActiveImage = ((System.Drawing.Image)(resources.GetObject("btnCloseAdd.ActiveImage")));
            this.btnCloseAdd.AllowAnimations = true;
            this.btnCloseAdd.AllowBuffering = false;
            this.btnCloseAdd.AllowToggling = false;
            this.btnCloseAdd.AllowZooming = true;
            this.btnCloseAdd.AllowZoomingOnFocus = false;
            this.btnCloseAdd.BackColor = System.Drawing.Color.Transparent;
            this.btnCloseAdd.DialogResult = System.Windows.Forms.DialogResult.None;
            this.btnCloseAdd.ErrorImage = ((System.Drawing.Image)(resources.GetObject("btnCloseAdd.ErrorImage")));
            this.btnCloseAdd.FadeWhenInactive = false;
            this.btnCloseAdd.Flip = Bunifu.UI.WinForms.BunifuImageButton.FlipOrientation.Normal;
            this.btnCloseAdd.ForeColor = System.Drawing.Color.Red;
            this.btnCloseAdd.Image = ((System.Drawing.Image)(resources.GetObject("btnCloseAdd.Image")));
            this.btnCloseAdd.ImageActive = ((System.Drawing.Image)(resources.GetObject("btnCloseAdd.ImageActive")));
            this.btnCloseAdd.ImageLocation = null;
            this.btnCloseAdd.ImageMargin = 20;
            this.btnCloseAdd.ImageSize = new System.Drawing.Size(28, 32);
            this.btnCloseAdd.ImageZoomSize = new System.Drawing.Size(48, 52);
            this.btnCloseAdd.InitialImage = ((System.Drawing.Image)(resources.GetObject("btnCloseAdd.InitialImage")));
            this.btnCloseAdd.Location = new System.Drawing.Point(569, 19);
            this.btnCloseAdd.Name = "btnCloseAdd";
            this.btnCloseAdd.Rotation = 0;
            this.btnCloseAdd.ShowActiveImage = true;
            this.btnCloseAdd.ShowCursorChanges = true;
            this.btnCloseAdd.ShowImageBorders = true;
            this.btnCloseAdd.ShowSizeMarkers = false;
            this.btnCloseAdd.Size = new System.Drawing.Size(48, 52);
            this.btnCloseAdd.TabIndex = 64;
            this.btnCloseAdd.ToolTipText = "";
            this.btnCloseAdd.WaitOnLoad = false;
            this.btnCloseAdd.Zoom = 20;
            this.btnCloseAdd.ZoomSpeed = 10;
            this.btnCloseAdd.Click += new System.EventHandler(this.btnCloseAdd_Click);
            // 
            // txbAddress_2
            // 
            this.txbAddress_2.BackColor = System.Drawing.Color.White;
            this.txbAddress_2.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.txbAddress_2.FormattingEnabled = true;
            this.txbAddress_2.ItemHeight = 24;
            this.txbAddress_2.Items.AddRange(new object[] {
            "Hà Nội",
            "Đà Đẵng",
            "Thừa Thiên Huế",
            "Hồ Chí Minh",
            "Hải Phòng",
            "Thanh Hóa",
            "Nghệ An",
            "Hà Tĩnh",
            "Quảng Bình",
            "Quảng Trị",
            "Quảng Nam",
            "Ninh Thuận",
            "Bình Thuận",
            "Khác"});
            this.txbAddress_2.Location = new System.Drawing.Point(332, 211);
            this.txbAddress_2.Name = "txbAddress_2";
            this.txbAddress_2.Size = new System.Drawing.Size(228, 30);
            this.txbAddress_2.Style = MetroFramework.MetroColorStyle.Green;
            this.txbAddress_2.TabIndex = 62;
            this.txbAddress_2.UseCustomBackColor = true;
            this.txbAddress_2.UseCustomForeColor = true;
            this.txbAddress_2.UseSelectable = true;
            this.txbAddress_2.UseStyleColors = true;
            // 
            // txbNationality_2
            // 
            this.txbNationality_2.BackColor = System.Drawing.Color.White;
            this.txbNationality_2.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.txbNationality_2.FormattingEnabled = true;
            this.txbNationality_2.ItemHeight = 24;
            this.txbNationality_2.Items.AddRange(new object[] {
            "Việt Nam",
            "Trung Quốc",
            "Hàn Quốc",
            "Nhật Bản",
            "Đài Loan",
            "Malaysia",
            "Thái Lan",
            "Singapore",
            "Nga",
            "Anh",
            "Pháp",
            "Đức",
            "Hoa Kỳ",
            "Hà Lan",
            "Tây Ban Nha",
            "Ý",
            "Khác"});
            this.txbNationality_2.Location = new System.Drawing.Point(332, 283);
            this.txbNationality_2.Name = "txbNationality_2";
            this.txbNationality_2.Size = new System.Drawing.Size(228, 30);
            this.txbNationality_2.Style = MetroFramework.MetroColorStyle.Green;
            this.txbNationality_2.TabIndex = 60;
            this.txbNationality_2.UseCustomBackColor = true;
            this.txbNationality_2.UseCustomForeColor = true;
            this.txbNationality_2.UseSelectable = true;
            this.txbNationality_2.UseStyleColors = true;
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Font = new System.Drawing.Font("Segoe UI", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label14.ForeColor = System.Drawing.Color.SeaGreen;
            this.label14.Location = new System.Drawing.Point(328, 110);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(115, 23);
            this.label14.TabIndex = 55;
            this.label14.Text = "Số điện thoại:";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Font = new System.Drawing.Font("Segoe UI", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label11.ForeColor = System.Drawing.Color.SeaGreen;
            this.label11.Location = new System.Drawing.Point(328, 180);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(66, 23);
            this.label11.TabIndex = 57;
            this.label11.Text = "Địa chỉ:";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Font = new System.Drawing.Font("Segoe UI", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label12.ForeColor = System.Drawing.Color.SeaGreen;
            this.label12.Location = new System.Drawing.Point(330, 255);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(88, 23);
            this.label12.TabIndex = 56;
            this.label12.Text = "Quốc tịch:";
            // 
            // comboBoxSex_2
            // 
            this.comboBoxSex_2.BackColor = System.Drawing.Color.White;
            this.comboBoxSex_2.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.comboBoxSex_2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(64)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
            this.comboBoxSex_2.FormattingEnabled = true;
            this.comboBoxSex_2.ItemHeight = 24;
            this.comboBoxSex_2.Items.AddRange(new object[] {
            "Nam",
            "Nữ",
            "Khác"});
            this.comboBoxSex_2.Location = new System.Drawing.Point(34, 211);
            this.comboBoxSex_2.Name = "comboBoxSex_2";
            this.comboBoxSex_2.Size = new System.Drawing.Size(243, 30);
            this.comboBoxSex_2.Style = MetroFramework.MetroColorStyle.Green;
            this.comboBoxSex_2.TabIndex = 4;
            this.comboBoxSex_2.UseCustomBackColor = true;
            this.comboBoxSex_2.UseCustomForeColor = true;
            this.comboBoxSex_2.UseSelectable = true;
            this.comboBoxSex_2.UseStyleColors = true;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Font = new System.Drawing.Font("Segoe UI", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label13.ForeColor = System.Drawing.Color.SeaGreen;
            this.label13.Location = new System.Drawing.Point(29, 183);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(79, 23);
            this.label13.TabIndex = 28;
            this.label13.Text = "Giới tính:";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Font = new System.Drawing.Font("Segoe UI", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label15.ForeColor = System.Drawing.Color.SeaGreen;
            this.label15.Location = new System.Drawing.Point(30, 108);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(89, 23);
            this.label15.TabIndex = 24;
            this.label15.Text = "Số CMND:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Segoe UI", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.Color.SeaGreen;
            this.label3.Location = new System.Drawing.Point(328, 41);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(90, 23);
            this.label3.TabIndex = 49;
            this.label3.Text = "Ngày sinh:";
            // 
            // txtFileName_2
            // 
            this.txtFileName_2.BorderColorFocused = System.Drawing.Color.White;
            this.txtFileName_2.BorderColorIdle = System.Drawing.Color.White;
            this.txtFileName_2.BorderColorMouseHover = System.Drawing.Color.White;
            this.txtFileName_2.BorderThickness = 1;
            this.txtFileName_2.characterCasing = System.Windows.Forms.CharacterCasing.Normal;
            this.txtFileName_2.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.txtFileName_2.Enabled = false;
            this.txtFileName_2.Font = new System.Drawing.Font("Century Gothic", 9.75F);
            this.txtFileName_2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(64)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
            this.txtFileName_2.isPassword = false;
            this.txtFileName_2.Location = new System.Drawing.Point(147, 40);
            this.txtFileName_2.Margin = new System.Windows.Forms.Padding(5);
            this.txtFileName_2.MaxLength = 32767;
            this.txtFileName_2.Name = "txtFileName_2";
            this.txtFileName_2.Size = new System.Drawing.Size(389, 36);
            this.txtFileName_2.TabIndex = 76;
            this.txtFileName_2.TextAlign = System.Windows.Forms.HorizontalAlignment.Left;
            // 
            // datepickerDateOfBirth
            // 
            this.datepickerDateOfBirth.BackColor = System.Drawing.Color.SeaGreen;
            this.datepickerDateOfBirth.BorderRadius = 0;
            this.datepickerDateOfBirth.ForeColor = System.Drawing.Color.White;
            this.datepickerDateOfBirth.Format = System.Windows.Forms.DateTimePickerFormat.Short;
            this.datepickerDateOfBirth.FormatCustom = null;
            this.datepickerDateOfBirth.Location = new System.Drawing.Point(332, 67);
            this.datepickerDateOfBirth.Margin = new System.Windows.Forms.Padding(66, 138, 66, 138);
            this.datepickerDateOfBirth.Name = "datepickerDateOfBirth";
            this.datepickerDateOfBirth.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.datepickerDateOfBirth.Size = new System.Drawing.Size(228, 30);
            this.datepickerDateOfBirth.TabIndex = 68;
            this.datepickerDateOfBirth.Value = new System.DateTime(2018, 2, 23, 23, 29, 57, 962);
            // 
            // btnBrowse_2
            // 
            this.btnBrowse_2.ActiveBorderThickness = 1;
            this.btnBrowse_2.ActiveCornerRadius = 1;
            this.btnBrowse_2.ActiveFillColor = System.Drawing.Color.SeaGreen;
            this.btnBrowse_2.ActiveForecolor = System.Drawing.Color.White;
            this.btnBrowse_2.ActiveLineColor = System.Drawing.Color.SeaGreen;
            this.btnBrowse_2.BackColor = System.Drawing.Color.White;
            this.btnBrowse_2.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("btnBrowse_2.BackgroundImage")));
            this.btnBrowse_2.ButtonText = "Browse...";
            this.btnBrowse_2.Cursor = System.Windows.Forms.Cursors.Hand;
            this.btnBrowse_2.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnBrowse_2.ForeColor = System.Drawing.Color.SeaGreen;
            this.btnBrowse_2.IdleBorderThickness = 1;
            this.btnBrowse_2.IdleCornerRadius = 1;
            this.btnBrowse_2.IdleFillColor = System.Drawing.Color.White;
            this.btnBrowse_2.IdleForecolor = System.Drawing.Color.SeaGreen;
            this.btnBrowse_2.IdleLineColor = System.Drawing.Color.SeaGreen;
            this.btnBrowse_2.Location = new System.Drawing.Point(17, 43);
            this.btnBrowse_2.Margin = new System.Windows.Forms.Padding(5);
            this.btnBrowse_2.Name = "btnBrowse_2";
            this.btnBrowse_2.Size = new System.Drawing.Size(120, 40);
            this.btnBrowse_2.TabIndex = 66;
            this.btnBrowse_2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.btnBrowse_2.Click += new System.EventHandler(this.btnBrowse_2_Click);
            // 
            // txtNumberPhone_2
            // 
            this.txtNumberPhone_2.BorderColorFocused = System.Drawing.Color.SeaGreen;
            this.txtNumberPhone_2.BorderColorIdle = System.Drawing.Color.SeaGreen;
            this.txtNumberPhone_2.BorderColorMouseHover = System.Drawing.Color.SeaGreen;
            this.txtNumberPhone_2.BorderThickness = 1;
            this.txtNumberPhone_2.characterCasing = System.Windows.Forms.CharacterCasing.Normal;
            this.txtNumberPhone_2.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.txtNumberPhone_2.Font = new System.Drawing.Font("Century Gothic", 9.75F);
            this.txtNumberPhone_2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(64)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
            this.txtNumberPhone_2.isPassword = false;
            this.txtNumberPhone_2.Location = new System.Drawing.Point(332, 138);
            this.txtNumberPhone_2.Margin = new System.Windows.Forms.Padding(5);
            this.txtNumberPhone_2.MaxLength = 9;
            this.txtNumberPhone_2.Name = "txtNumberPhone_2";
            this.txtNumberPhone_2.Size = new System.Drawing.Size(228, 36);
            this.txtNumberPhone_2.TabIndex = 75;
            this.txtNumberPhone_2.TextAlign = System.Windows.Forms.HorizontalAlignment.Left;
            // 
            // txtCMND_2
            // 
            this.txtCMND_2.BorderColorFocused = System.Drawing.Color.SeaGreen;
            this.txtCMND_2.BorderColorIdle = System.Drawing.Color.SeaGreen;
            this.txtCMND_2.BorderColorMouseHover = System.Drawing.Color.SeaGreen;
            this.txtCMND_2.BorderThickness = 1;
            this.txtCMND_2.characterCasing = System.Windows.Forms.CharacterCasing.Normal;
            this.txtCMND_2.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.txtCMND_2.Font = new System.Drawing.Font("Century Gothic", 9.75F);
            this.txtCMND_2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(64)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
            this.txtCMND_2.isPassword = false;
            this.txtCMND_2.Location = new System.Drawing.Point(34, 139);
            this.txtCMND_2.Margin = new System.Windows.Forms.Padding(5);
            this.txtCMND_2.MaxLength = 9;
            this.txtCMND_2.Name = "txtCMND_2";
            this.txtCMND_2.Size = new System.Drawing.Size(246, 39);
            this.txtCMND_2.TabIndex = 74;
            this.txtCMND_2.TextAlign = System.Windows.Forms.HorizontalAlignment.Left;
            // 
            // txtFullName_2
            // 
            this.txtFullName_2.BorderColorFocused = System.Drawing.Color.SeaGreen;
            this.txtFullName_2.BorderColorIdle = System.Drawing.Color.SeaGreen;
            this.txtFullName_2.BorderColorMouseHover = System.Drawing.Color.SeaGreen;
            this.txtFullName_2.BorderThickness = 1;
            this.txtFullName_2.characterCasing = System.Windows.Forms.CharacterCasing.Normal;
            this.txtFullName_2.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.txtFullName_2.Font = new System.Drawing.Font("Century Gothic", 9.75F);
            this.txtFullName_2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(64)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
            this.txtFullName_2.isPassword = false;
            this.txtFullName_2.Location = new System.Drawing.Point(34, 67);
            this.txtFullName_2.Margin = new System.Windows.Forms.Padding(5);
            this.txtFullName_2.MaxLength = 32767;
            this.txtFullName_2.Name = "txtFullName_2";
            this.txtFullName_2.Size = new System.Drawing.Size(246, 36);
            this.txtFullName_2.TabIndex = 73;
            this.txtFullName_2.TextAlign = System.Windows.Forms.HorizontalAlignment.Left;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.txtFileName_2);
            this.groupBox2.Controls.Add(this.btnBrowse_2);
            this.groupBox2.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.groupBox2.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.groupBox2.ForeColor = System.Drawing.Color.SeaGreen;
            this.groupBox2.Location = new System.Drawing.Point(16, 337);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(544, 110);
            this.groupBox2.TabIndex = 70;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Voice Audio";
            // 
            // groupCustomer
            // 
            this.groupCustomer.Controls.Add(this.txtNumberPhone_2);
            this.groupCustomer.Controls.Add(this.txtCMND_2);
            this.groupCustomer.Controls.Add(this.txtFullName_2);
            this.groupCustomer.Controls.Add(this.groupBox2);
            this.groupCustomer.Controls.Add(this.datepickerDateOfBirth);
            this.groupCustomer.Controls.Add(this.txbAddress_2);
            this.groupCustomer.Controls.Add(this.txbNationality_2);
            this.groupCustomer.Controls.Add(this.label14);
            this.groupCustomer.Controls.Add(this.label11);
            this.groupCustomer.Controls.Add(this.label12);
            this.groupCustomer.Controls.Add(this.comboBoxSex_2);
            this.groupCustomer.Controls.Add(this.label13);
            this.groupCustomer.Controls.Add(this.label15);
            this.groupCustomer.Controls.Add(this.label3);
            this.groupCustomer.Controls.Add(this.label16);
            this.groupCustomer.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.groupCustomer.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.groupCustomer.ForeColor = System.Drawing.Color.SeaGreen;
            this.groupCustomer.Location = new System.Drawing.Point(19, 99);
            this.groupCustomer.Name = "groupCustomer";
            this.groupCustomer.Size = new System.Drawing.Size(598, 463);
            this.groupCustomer.TabIndex = 63;
            this.groupCustomer.TabStop = false;
            this.groupCustomer.Text = "Thông tin đối tượng";
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Font = new System.Drawing.Font("Segoe UI", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label16.ForeColor = System.Drawing.Color.SeaGreen;
            this.label16.Location = new System.Drawing.Point(30, 38);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(88, 23);
            this.label16.TabIndex = 22;
            this.label16.Text = "Họ và tên:";
            // 
            // bunifuSeparator1
            // 
            this.bunifuSeparator1.BackColor = System.Drawing.Color.Transparent;
            this.bunifuSeparator1.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(224)))), ((int)(((byte)(224)))), ((int)(((byte)(224)))));
            this.bunifuSeparator1.LineThickness = 1;
            this.bunifuSeparator1.Location = new System.Drawing.Point(10, 79);
            this.bunifuSeparator1.Margin = new System.Windows.Forms.Padding(5);
            this.bunifuSeparator1.Name = "bunifuSeparator1";
            this.bunifuSeparator1.Size = new System.Drawing.Size(604, 12);
            this.bunifuSeparator1.TabIndex = 61;
            this.bunifuSeparator1.Transparency = 255;
            this.bunifuSeparator1.Vertical = false;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Segoe UI", 20.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.SeaGreen;
            this.label2.Location = new System.Drawing.Point(6, 32);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(271, 46);
            this.label2.TabIndex = 60;
            this.label2.Text = "Thêm Đối Tượng";
            // 
            // btnClose_2
            // 
            this.btnClose_2.ActiveBorderThickness = 1;
            this.btnClose_2.ActiveCornerRadius = 20;
            this.btnClose_2.ActiveFillColor = System.Drawing.Color.SeaGreen;
            this.btnClose_2.ActiveForecolor = System.Drawing.Color.White;
            this.btnClose_2.ActiveLineColor = System.Drawing.Color.SeaGreen;
            this.btnClose_2.BackColor = System.Drawing.Color.White;
            this.btnClose_2.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("btnClose_2.BackgroundImage")));
            this.btnClose_2.ButtonText = "Đóng";
            this.btnClose_2.Cursor = System.Windows.Forms.Cursors.Hand;
            this.btnClose_2.Font = new System.Drawing.Font("Segoe UI", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnClose_2.ForeColor = System.Drawing.Color.SeaGreen;
            this.btnClose_2.IdleBorderThickness = 1;
            this.btnClose_2.IdleCornerRadius = 20;
            this.btnClose_2.IdleFillColor = System.Drawing.Color.White;
            this.btnClose_2.IdleForecolor = System.Drawing.Color.SeaGreen;
            this.btnClose_2.IdleLineColor = System.Drawing.Color.SeaGreen;
            this.btnClose_2.Location = new System.Drawing.Point(320, 30);
            this.btnClose_2.Margin = new System.Windows.Forms.Padding(5, 6, 5, 6);
            this.btnClose_2.Name = "btnClose_2";
            this.btnClose_2.Size = new System.Drawing.Size(271, 49);
            this.btnClose_2.TabIndex = 51;
            this.btnClose_2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // btnAddObject_2
            // 
            this.btnAddObject_2.ActiveBorderThickness = 1;
            this.btnAddObject_2.ActiveCornerRadius = 20;
            this.btnAddObject_2.ActiveFillColor = System.Drawing.Color.SeaGreen;
            this.btnAddObject_2.ActiveForecolor = System.Drawing.Color.White;
            this.btnAddObject_2.ActiveLineColor = System.Drawing.Color.SeaGreen;
            this.btnAddObject_2.BackColor = System.Drawing.Color.White;
            this.btnAddObject_2.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("btnAddObject_2.BackgroundImage")));
            this.btnAddObject_2.ButtonText = "Thêm";
            this.btnAddObject_2.Cursor = System.Windows.Forms.Cursors.Hand;
            this.btnAddObject_2.Font = new System.Drawing.Font("Segoe UI", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnAddObject_2.ForeColor = System.Drawing.Color.SeaGreen;
            this.btnAddObject_2.IdleBorderThickness = 1;
            this.btnAddObject_2.IdleCornerRadius = 20;
            this.btnAddObject_2.IdleFillColor = System.Drawing.Color.White;
            this.btnAddObject_2.IdleForecolor = System.Drawing.Color.SeaGreen;
            this.btnAddObject_2.IdleLineColor = System.Drawing.Color.SeaGreen;
            this.btnAddObject_2.Location = new System.Drawing.Point(16, 30);
            this.btnAddObject_2.Margin = new System.Windows.Forms.Padding(5, 6, 5, 6);
            this.btnAddObject_2.Name = "btnAddObject_2";
            this.btnAddObject_2.Size = new System.Drawing.Size(271, 49);
            this.btnAddObject_2.TabIndex = 48;
            this.btnAddObject_2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.btnAddObject_2.Click += new System.EventHandler(this.btnAddObject_2_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnClose_2);
            this.groupBox1.Controls.Add(this.btnAddObject_2);
            this.groupBox1.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.groupBox1.ForeColor = System.Drawing.Color.SeaGreen;
            this.groupBox1.Location = new System.Drawing.Point(12, 592);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox1.Size = new System.Drawing.Size(604, 101);
            this.groupBox1.TabIndex = 62;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Chức năng";
            // 
            // fAddObject
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(622, 713);
            this.Controls.Add(this.btnCloseAdd);
            this.Controls.Add(this.groupCustomer);
            this.Controls.Add(this.bunifuSeparator1);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
            this.Name = "fAddObject";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "fAddObject";
            this.groupBox2.ResumeLayout(false);
            this.groupCustomer.ResumeLayout(false);
            this.groupCustomer.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Bunifu.UI.WinForms.BunifuImageButton btnCloseAdd;
        private MetroFramework.Controls.MetroComboBox txbAddress_2;
        private MetroFramework.Controls.MetroComboBox txbNationality_2;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label12;
        private MetroFramework.Controls.MetroComboBox comboBoxSex_2;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Label label3;
        private Bunifu.Framework.UI.BunifuMetroTextbox txtFileName_2;
        private Bunifu.Framework.UI.BunifuDatepicker datepickerDateOfBirth;
        private Bunifu.Framework.UI.BunifuThinButton2 btnBrowse_2;
        private Bunifu.Framework.UI.BunifuMetroTextbox txtNumberPhone_2;
        private Bunifu.Framework.UI.BunifuMetroTextbox txtCMND_2;
        private Bunifu.Framework.UI.BunifuMetroTextbox txtFullName_2;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.GroupBox groupCustomer;
        private System.Windows.Forms.Label label16;
        private Bunifu.Framework.UI.BunifuSeparator bunifuSeparator1;
        private System.Windows.Forms.Label label2;
        private Bunifu.Framework.UI.BunifuThinButton2 btnClose_2;
        private Bunifu.Framework.UI.BunifuThinButton2 btnAddObject_2;
        private System.Windows.Forms.GroupBox groupBox1;
    }
}