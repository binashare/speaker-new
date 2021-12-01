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
using NAudio.Wave;

namespace HotelManagerSystem.User_Control
{
    public partial class ucManageRoom : UserControl
    {
        string path;
        private WaveFileWriter RecordedAudioWriter = null;
        private WasapiLoopbackCapture CaptureInstance = null;

        public ucManageRoom()
        {
            InitializeComponent();
            btnRecorder.Visible = true;
            btnStop_Recorder.Visible = false;
            display_2.Visible = true;
            stop_2.Visible = false;
        }
        System.Diagnostics.Stopwatch count = new System.Diagnostics.Stopwatch();
        private void btnRecorder_Click(object sender, EventArgs e)
        {
            btnRecorder.Visible = false;
            btnStop_Recorder.Visible = true;
            timer1.Start();
            count.Start();
            groupBox2.Text = "Recording...";
            groupBox2.ForeColor = System.Drawing.Color.Red;
            string outputFilePath = @"D:\system_recorded_audio.wav";
            this.CaptureInstance = new WasapiLoopbackCapture();
            this.RecordedAudioWriter = new WaveFileWriter(outputFilePath, CaptureInstance.WaveFormat);
            this.CaptureInstance.DataAvailable += (s, a) =>
            {
                this.RecordedAudioWriter.Write(a.Buffer, 0, a.BytesRecorded);
            };
            this.CaptureInstance.RecordingStopped += (s, a) =>
            {
                this.RecordedAudioWriter.Dispose();
                this.RecordedAudioWriter = null;
                CaptureInstance.Dispose();
            };
            this.CaptureInstance.StartRecording();
        }

        private void btnStop_Recorder_Click(object sender, EventArgs e)
        {
            timer1.Stop();
            count.Stop();
            btnRecorder.Visible = true;
            btnStop_Recorder.Visible = false;
            this.CaptureInstance.StopRecording();
            groupBox2.Text = "Real Time";
            groupBox2.ForeColor = System.Drawing.Color.SeaGreen;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            TimeSpan elapsed = count.Elapsed;
            labelTimer.Text = string.Format("{0:00}:{1:00}:{2:00}", Math.Floor(elapsed.TotalHours), elapsed.Minutes, elapsed.Seconds);
        }

        private void btnThoatRecog_Click(object sender, EventArgs e)
        {
            ucHome uchome = new ucHome();
            uchome.Dock = DockStyle.Fill;
            fHome.frmHome.MetroContainer.Controls.Add(uchome);
            fHome.frmHome.MetroContainer.Controls["ucHome"].BringToFront();
            foreach (ucManageRoom uc in fHome.frmHome.MetroContainer.Controls.OfType<ucManageRoom>())
            {
                fHome.frmHome.MetroContainer.Controls.Remove(uc);
            }
        }

        private void label_Timer_Click(object sender, EventArgs e)
        {

        }
    }
}
