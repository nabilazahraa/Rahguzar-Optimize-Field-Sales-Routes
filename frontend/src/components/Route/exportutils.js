// src/utils/exportUtils.js
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';
import { format, parseISO } from 'date-fns';

const formatDateWithDay = (dateStr) => {
  const date = parseISO(dateStr);
  return format(date, 'EEEE, dd/MM');
};

export const exportScheduleToExcel = (pjp, monthly_pjp, isMonthly) => {
  const allData = [];
  const sourcePjp = isMonthly ? monthly_pjp : pjp;

  Object.keys(sourcePjp).forEach((orderbookerId) => {
    Object.keys(sourcePjp[orderbookerId]).forEach((day) => {
      const stores = sourcePjp[orderbookerId][day].map((store, index) => ({
        Orderbooker: orderbookerId,
        Day: isMonthly ? formatDateWithDay(day) : day,
        ID: index + 1,
        StoreID: store.storeid,
        StoreCode: store.storecode,
        Latitude: store.latitude.toFixed(4),
        Longitude: store.longitude.toFixed(4),
      }));
      allData.push(...stores);
    });
  });

  const worksheet = XLSX.utils.json_to_sheet(allData);
  const workbook = {
    Sheets: { All_Results: worksheet },
    SheetNames: ['All_Results'],
  };

  const excelBuffer = XLSX.write(workbook, {
    bookType: 'xlsx',
    type: 'array',
  });

  const blob = new Blob([excelBuffer], {
    type:
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8',
  });

  saveAs(blob, 'Schedule.xlsx');
};
