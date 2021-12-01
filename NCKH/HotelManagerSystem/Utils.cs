using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace HotelManagerSystem
{
    public class Utils
    {
        public bool CheckFillInText(Control[] controls)
        {
            foreach(var control in controls)
            {
                if(control.Name == string.Empty)
                {
                    return false;
                }
            }
            return true;
        }
        public void Trim(Bunifu.UI.WinForms.BunifuTextBox[] textBoxes)
        {
            foreach(var textBox in textBoxes)
            {
                textBox.Text = textBox.Text.Trim();
            }
        }
        public string ConvertFormatString(string text)
        {
            if (text.Contains(".") || text.Contains(" "))
            {
                string[] vs = text.Split(new char[] { '.', ' ' });
                StringBuilder textNow = new StringBuilder();
                for (int i = 0; i < vs.Length - 1; i++)
                {
                    textNow.Append(vs[i]);
                }
                return textNow.ToString();
            }
            else return text;
        }
        public string ConvertStringToInt(string text)
        {
            if(text.Contains(".") || text.Contains(" "))
            {
                string[] vs = text.Split(new char[] { '.', ' ' });
                StringBuilder textNow = new StringBuilder();
                for(int i = 0; i < vs.Length - 1; i++)
                {
                    textNow.Append(vs[i]);
                }
                return textNow.ToString();
            }
            return text;
        }
        public string ConvertFormatCulture(string text)
        {
            if (text == string.Empty)
                return 0.ToString("C0", CultureInfo.CreateSpecificCulture("vi-VN"));
            if (text.Contains(".") || text.Contains(" "))
                return text;
            else
                return (int.Parse(text).ToString("C0", CultureInfo.CreateSpecificCulture("vi-VN")));
        }
    }
}
