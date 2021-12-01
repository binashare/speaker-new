using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HotelManagerSystem.DAO
{
    class DataProvider
    {
        public static SqlConnection KetNoi()
        {
            string sChuoiKetNoi = @"Data Source=LAPTOP-95RTQ1ED\SQLEXPRESS; Initial Catalog=LISTOBJECT; Integrated Security=True";
            SqlConnection conn = new SqlConnection(sChuoiKetNoi);
            conn.Open();
            return conn;
        }

        public static DataTable TruyVanDataReader(string sChuoiTruyVan)
        {
            try
            {
                SqlCommand cmd = new SqlCommand(sChuoiTruyVan, KetNoi());
                SqlDataReader rd = cmd.ExecuteReader();
                DataTable dt = new DataTable();
                dt.Load(rd);
                KetNoi().Close();
                return dt;

            }
            catch (Exception ex)
            {
                return null;
            }
        }
        public static bool TruyVanExecuteNonQuery(string sChuoiTruyVan)
        {
            try
            {
                SqlCommand cmd = new SqlCommand(sChuoiTruyVan, KetNoi());
                int iCMD = cmd.ExecuteNonQuery();
                KetNoi().Close();
                if (iCMD > 0)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            catch (Exception ex)
            {
                KetNoi().Close();
                return false;
            }
        }
    }
}
