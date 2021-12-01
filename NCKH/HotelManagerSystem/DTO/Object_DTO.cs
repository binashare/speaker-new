using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MongoDB.Driver;
using MongoDB.Bson;
namespace HotelManagerSystem.DTO
{
    public class Object_DTO
    {
     
        public string TenDT { set; get; }
        public string NgaySinh { set; get; }
        public string GioiTinh { set; get; }
        public string PathFile { set; get; }
        public string CMND { set; get; }
        public string Phone { set; get; }
        public string QuocTich { set; get; }
        public string QueQuan { set; get; }
        public float score { set; get; }
        public BsonArray Feature { set; get; }
        public ObjectId _id { set; get; }

        public Object_DTO()
        {

        }
        public Object_DTO(DataRow row)
        {
            this._id = (ObjectId)row["_id"];
            this.TenDT = row["TenDT"].ToString();
            this.CMND = row["CMND"].ToString();
            string a = row["NgaySinh"].ToString().Trim();
            this.NgaySinh = (DateTime.Parse(a)).ToString("yyyy-MM-dd");
            this.Phone = row["Phone"].ToString();
            this.QuocTich = row["QuocTich"].ToString();
            this.QueQuan = row["QueQuan"].ToString();
            this.GioiTinh = row["GioiTinh"].ToString();
            this.PathFile = row["pathFile"].ToString();
            this.Feature = (BsonArray)row["Feature"].ToString();

        }
        public bool Equals(Object_DTO objectPre)
        {

            if (this.TenDT != objectPre.TenDT) return false;
            if (this.NgaySinh != objectPre.NgaySinh) return false;
            if (this.Phone != objectPre.Phone) return false;
            if (this.QueQuan != objectPre.QueQuan) return false;
            if (this.QuocTich != objectPre.QuocTich) return false;
            if (this.PathFile != objectPre.PathFile) return false;
            if (this.GioiTinh != objectPre.GioiTinh) return false;
            return true;
        }
    }
}
