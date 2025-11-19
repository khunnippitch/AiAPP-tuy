let model;

// โหลดโมเดลที่ฝึกไว้แล้ว (โมเดลจำแนกหมาแมว)
async function loadModel() {
    // ใช้ path สัมพัทธ์ไปยังไฟล์ model.json ของคุณ
    const modelURL = "./model.json"; 
    
    try {
        model = await tf.loadLayersModel(modelURL);
        console.log("✅ โมเดลโหลดเสร็จแล้ว");
        document.getElementById("result").innerText = "พร้อมใช้งาน! กรุณาเลือกรูปภาพ";
    } catch (error) {
        console.error("❌ เกิดข้อผิดพลาดในการโหลดโมเดล:", error);
        document.getElementById("result").innerText = "ข้อผิดพลาด: โหลดโมเดลไม่ได้! (ตรวจสอบไฟล์ model.json)";
    }
}

// แสดงภาพ preview เมื่ออัปโหลด
document.getElementById("imageInput").addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (event) {
        const img = document.getElementById("preview");
        img.src = event.target.result;
        img.hidden = false;
        document.getElementById("result").innerText = "กด 'วิเคราะห์ภาพ' เพื่อเริ่ม";
    };
    reader.readAsDataURL(file);
});

// ปุ่มวิเคราะห์ภาพ
document.getElementById("predictBtn").addEventListener("click", async () => {
    const imgElement = document.getElementById("preview");
    
    if (!model) {
        alert("กรุณารอให้โมเดลโหลดก่อนนะครับ...");
        return;
    }
    if (imgElement.hidden) {
        alert("กรุณาเลือกรูปภาพก่อนครับ");
        return;
    }
    
    document.getElementById("result").innerText = "กำลังวิเคราะห์...";

    // แปลงภาพเป็น tensor, ปรับขนาด (224x224 ตามที่ Teachable Machine กำหนด)
    // และทำให้เป็นค่า Float และเพิ่มมิติ Batch 
    const tensor = tf.browser.fromPixels(imgElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(127.5) // ปรับช่วงค่าให้เป็น [-1, 1]
        .sub(1)
        .expandDims();

    // ทำนายผล
    const prediction = await model.predict(tensor).data();

    // 🔴 ส่วนที่แก้ไข: สลับตำแหน่งการดึงค่าจาก Array
    // สมมติว่า Index 0 คือ Dog และ Index 1 คือ Cat เพื่อแก้ปัญหาที่ทายผิด
    const dogConfidence = prediction[0]; // ใช้ Index 0 สำหรับ Dog
    const catConfidence = prediction[1]; // ใช้ Index 1 สำหรับ Cat

    // เปรียบเทียบผลลัพธ์
    const resultText = dogConfidence > catConfidence
        ? `🐶 นี่คือ **หมา** (${(dogConfidence * 100).toFixed(2)}%)`
        : `🐱 นี่คือ **แมว** (${(catConfidence * 100).toFixed(2)}%)`;

    document.getElementById("result").innerHTML = resultText;
    
    // Cleanup
    tensor.dispose(); 
});

// โหลดโมเดลเมื่อเปิดหน้า
loadModel();