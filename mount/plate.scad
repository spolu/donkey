use <MCAD/boxes.scad>
use <fourHoles.scad>

plateH=5;
plateW=200;
plateL=240;

platePinW = 2.7; // actual mount hole diameter
platePinH = 3;
standoffFN = 30;

//PWM Dimensions
PWMspacingX = 56;
PWMspacingY = 19;
//rPi Dimensions
rPispacingX = 49;
rPispacingY = 68;
//Jetson Dimensions
jetsonspacingY = 159;
jetsonLeftX = -80;
jetsonFrontRightX = 76.5;
jetsonBackRightX = 54;
jetsonPinW = 2.7;
jetsonPinH = 8;
//RCCar Dimensions
carspacingX = 25;
carspacingY = 175;
carPinW = 5.8;
carPinClearanceW = 11;
carPinClearanceH = 2.5; //plate must me 3mm max for pins to work
//Handle Dimensions
handlePinSpacingX = plateW - 2 * 5;
handlePinSpacingY = plateL - 2 * 10;
handlePinW = 3.1;
//Meccano Dimensions0
meccanoPinW = 4.1;
meccanoSpacing = 12.7;

donkeyPlate();

module donkeyPlate () {
    difference() {
      union() {
            newrPiPlate(plateH, plateW, plateL);
            translate([0,20,0]) rPiMounts();
            translate([0,-(plateL-PWMspacingY)/2+10,0]) PWMMounts();
            translate([0,(plateL-jetsonspacingY)/2-40,0])jetsonMounts();
          }
      translate([0,(plateL-jetsonspacingY)/2-40,0]) jetsonHoles();
      translate([0,20,0]) rPiHoles();
      translate([0,-(plateL-PWMspacingY)/2+10,0]) PWMHoles();
      meccanoHoles();
      handleHoles();
      frontCameraMountingHoles();
  }
}

module newrPiPlate(piPlateH=3, piPlateW=95, piPlateL=195) {
    difference() {
      union(){
        newPlate(piPlateH, piPlateW, piPlateL);
      }
      #translate([0,(plateL-carspacingY)/2-20,0]) carHoles(); //not more than 2cm between these holes and front border so the bumper can protect the car
      #translate([0,(plateL-carspacingY)/2-20,0]) carPinClearance();
      #translate([0,-40,0])PWMCommandHoles();
      #translate([0,20,0]) batteryStrapHoles();
    }
}

module newPlate (piPlateH=5, piPlateW=95, piPlateL=195) {
    translate([0,0,plateH/2])
    roundedBox([piPlateW,piPlateL,piPlateH],3, true);   
}

module rPiHoles () {
    fourHoles (rPispacingX, rPispacingY, platePinW, plateH*4, standoffFN, plateH);
}

module PWMHoles () {
    fourHoles (PWMspacingX, PWMspacingY, platePinW, plateH*4, standoffFN, plateH);
}

module PWMCommandHoles () {
    roundedBox([50,20,3*plateH],3, true);   
}

module batteryStrapHoles () {
    spacingFromPi = 15;
    translate([0, rPispacingY/2+spacingFromPi,0]) roundedBox([30,5,3*plateH],3, true);
    translate([rPispacingX/2+spacingFromPi,0,0]) roundedBox([5,30,3*plateH],3, true);
    translate([-rPispacingX/2-spacingFromPi,0,0]) roundedBox([5,30,3*plateH],3, true);
}

module carHoles () {
    fourHoles (carspacingX, carspacingY, carPinW, plateH*4, standoffFN, plateH);     
}

module carPinClearance () {
    translate([0,0,-carPinClearanceH-plateH]) fourMounts (carspacingX, carspacingY, carPinClearanceW, carPinClearanceH+1, standoffFN, plateH);
}

module handleHoles () {
    fourHoles (handlePinSpacingX, handlePinSpacingY, handlePinW, plateH*4, standoffFN, plateH);
}

module frontCameraMountingHoles(plateH=5) {
    $fn=30;
    plateH=15.0;
    platePinH=plateH;
    translate([8,plateL/2 - 7,plateH/2]) cylinder(r=platePinW/2, h=platePinH*5,center=true);
    translate([-8,plateL/2 - 7,plateH/2]) cylinder(r=platePinW/2, h=platePinH*5,center=true);
}


module meccanoHoles() {
    for (i = [0:7]) {
            fourHoles (plateW-meccanoSpacing, (2*i+1)*meccanoSpacing, meccanoPinW, plateH*4, standoffFN, plateH);
        }
}


module jetsonHoles () {
    $fn=standoffFN;
    translate([jetsonFrontRightX,jetsonspacingY/2,plateH/2]) cylinder(r=jetsonPinW/2,h=plateH*10,center=true);
    translate([jetsonLeftX,jetsonspacingY/2,plateH/2]) rotate([0,0,90]) cylinder(r=jetsonPinW/2,h=plateH*10,center=true);
    translate([jetsonBackRightX,-jetsonspacingY/2,plateH/2]) rotate([0,0,90])cylinder(r=jetsonPinW/2,h=plateH*10,center=true);
    translate([jetsonLeftX,-jetsonspacingY/2,plateH/2]) rotate([0,0,90])cylinder(r=jetsonPinW/2,h=plateH*10,center=true);    
}


module rPiMounts () {
    fourMounts (rPispacingX, rPispacingY, platePinW*2, platePinH, standoffFN, plateH);
}

module PWMMounts () {
    fourMounts (PWMspacingX, PWMspacingY, platePinW*2, platePinH, standoffFN, plateH);
}

module jetsonMounts () {
    $fn=standoffFN;
    mountDiameter= jetsonPinW*3;

    translate([jetsonFrontRightX,jetsonspacingY/2,plateH + jetsonPinH/2]) cylinder(r=mountDiameter/2,h=jetsonPinH,center=true);
    translate([jetsonLeftX,jetsonspacingY/2,plateH + jetsonPinH/2]) rotate([0,0,90]) cylinder(r=mountDiameter/2,h=jetsonPinH,center=true);
    translate([jetsonBackRightX,-jetsonspacingY/2,plateH + jetsonPinH/2]) rotate([0,0,90])cylinder(r=mountDiameter/2,h=jetsonPinH,center=true);
    translate([jetsonLeftX,-jetsonspacingY/2,plateH + jetsonPinH/2]) rotate([0,0,90])cylinder(r=mountDiameter/2,h=jetsonPinH,center=true);    
}
