using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HotelManagerSystem.DTO;
namespace HotelManagerSystem.DAO
{
    public class Object_DAL
    {
        public static DataTable LoadObject()
        {
            string sChuoiTruyVan = @"Select * from Object";
            DataTable dt = new DataTable();
            dt = DataProvider.TruyVanDataReader(sChuoiTruyVan);
            return dt;
        }
        public static bool InsertObject(Object_DTO obj_DTO)
        {
            string sChuoiTruyVan = string.Format(@"Insert into Object(MaDT,TenDT,NgaySinh,GioiTinh,CMND,Phone,QuocTich,QueQuan,pathFile)
                                                    Values ('{0}',N'{1}','{2}',N'{3}',N'{4}','{5}',N'{6}',N'{7}',N'{8}')",
                                                     obj_DTO.TenDT, obj_DTO.NgaySinh, obj_DTO.GioiTinh, obj_DTO.CMND, obj_DTO.Phone, obj_DTO.QuocTich, obj_DTO.QueQuan, obj_DTO.PathFile);
            bool ketqua = DataProvider.TruyVanExecuteNonQuery(sChuoiTruyVan);
            return ketqua;
        }
        public static bool UpdateObject(Object_DTO obj_DTO)
        {
            string sChuoiTruyVan = string.Format(@"update Object set TenDT=N'{0}', NgaySinh='{1}',GioiTinh=N'{2}',CMND='{3}',Phone='{4}',
                                                   QuocTich=N'{5}',QueQuan=N'{6}',pathFile=N'{7}' where MaDT='{8}'",
                                                   obj_DTO.TenDT, obj_DTO.NgaySinh, obj_DTO.GioiTinh, obj_DTO.CMND, obj_DTO.Phone,
                                                   obj_DTO.QuocTich, obj_DTO.QueQuan, obj_DTO.PathFile);
            bool ketqua = DataProvider.TruyVanExecuteNonQuery(sChuoiTruyVan);
            return ketqua;
        }
        public static bool RemoveObject(String MaDT)
        {
            string sChuoiTruyVan = string.Format(@"delete from Object where MaDT='{0}'", MaDT);
            bool ketqua = DataProvider.TruyVanExecuteNonQuery(sChuoiTruyVan);
            return ketqua;
        }
    }
}
