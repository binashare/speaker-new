using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HotelManagerSystem.DTO;
using HotelManagerSystem.DAO;
namespace HotelManagerSystem.BUL
{
    public class Object_BUL
    {
        public static DataTable LoadObject()
        {
            return Object_DAL.LoadObject();
        }
        public static bool InsertObject(Object_DTO obj_DTO)
        {
            return Object_DAL.InsertObject(obj_DTO);
        }
        public static bool UpdateObject(Object_DTO obj_DTO)
        {
            return Object_DAL.UpdateObject(obj_DTO);
        }
        public static bool RemoveObject(string MaDT)
        {
            return Object_DAL.RemoveObject(MaDT);
        }
    }
}
