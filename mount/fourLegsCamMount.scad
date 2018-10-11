use <fourHoles.scad>
use <MCAD/boxes.scad>

// A Raspberry Pi camera mount.
%import("plate.stl");

// Holes resolution
standoffFN = 30;

// Dimensions of the camera board.
cameraWidth = 38;
cameraHeight = 38;
aperatureWidth = 22;
aperatureYOffset = 5.5;
cameraBoardThickness = 1;

holeYOffset = 9.5;
holeXSpacing = 28;
holeYSpacing = 28;
holeDiameter = 2;

standoffDiameter = 6;
standoffHeight = 3;

mountMargin = 3;
mountBottomMargin = 10; // To accommodate the cable.
mountWidth = cameraWidth + 2*mountMargin;
mountHeight = cameraHeight + mountMargin + mountBottomMargin;

mountThickness = 3;

maxModelSize = 30;

//Handle Dimensions
plateH=5;
plateW=200;
plateL=240;
structureWidth = 10;
structureLength = 20;
structureHeight = 107;
structureBeamHeight = 15;
nutInsertionHeight = 3;

//Meccano Dimensions0
meccanoPinW = 4.1;
meccanoSpacing = 12.7;

handlePinSpacingX = plateW - 2 * structureWidth/2;
handlePinSpacingY = plateL - 2 * structureLength/2;
handlePinW = 3.1;

 camMount();
//camMount();

module camMount() {
    translate([0,0,5]) mainStructure();
    translate([0.3,99.5,62]) rotate([0,0,-90]) rotate([0,-83.5-21.5,0]) union() {
        mainMount();
        cameraBoard();
        }
 }

// The camera board.
module cameraBoard(){
%translate([mountBottomMargin, -cameraWidth/2, mountThickness+standoffHeight]) cube(size=[cameraHeight, cameraWidth, cameraBoardThickness]);
}

// The main part of the mount.
module mainMount() {
difference() {
  union() {
  translate([0, -mountWidth/2, 0])
  cube(size=[cameraHeight + mountMargin + mountBottomMargin,
             mountWidth,
             mountThickness]);
  // mount standoffs      
  translate([holeYOffset + mountBottomMargin + holeYSpacing/2, 0, 0]) fourMounts (holeYSpacing, holeXSpacing, standoffDiameter, standoffHeight, standoffFN, mountThickness);
  }
  
  // mounting holes
  translate([holeYOffset + mountBottomMargin + holeYSpacing/2, 0, 0]) fourHoles(holeYSpacing, holeXSpacing, holeDiameter, 2*maxModelSize, standoffFN, mountThickness); 
}
}

module mainStructure() {
    difference () {
        union() {
            //4 pilars
            translate([handlePinSpacingX/2, handlePinSpacingY/2, structureHeight/2]) roundedBox([structureWidth,structureLength,structureHeight],3, true);
            translate([-handlePinSpacingX/2, handlePinSpacingY/2, structureHeight/2]) roundedBox([structureWidth,structureLength,structureHeight],3, true);
            translate([handlePinSpacingX/2, -handlePinSpacingY/2, structureHeight/2]) roundedBox([structureWidth,structureLength,structureHeight],3, true);
            translate([-handlePinSpacingX/2, -handlePinSpacingY/2, structureHeight/2]) roundedBox([structureWidth,structureLength,structureHeight],3, true);
            
            //3 beams
            translate([0, handlePinSpacingY/2, structureHeight - structureBeamHeight/2]) roundedBox([plateW,structureLength,structureBeamHeight],3, true);
            translate([0, handlePinSpacingY/2 - structureLength/2, structureHeight - 5/2 ]) roundedBox([plateW,2*structureLength,5],3, true);
            translate([0, -handlePinSpacingY/2, structureHeight - structureBeamHeight/2]) roundedBox([plateW,structureLength,structureBeamHeight],3, true);
            translate([0, 0, structureHeight - structureBeamHeight/2]) roundedBox([1.5*structureWidth,plateL,structureBeamHeight],3, true);
        }
        
        translate([0.3,99.5,62]) rotate([0,0,-90]) rotate([0,-83.5-21.5,0]) translate([mountMargin, -mountWidth/2, 3]) cube(size=[cameraHeight + 3 * mountMargin + mountBottomMargin, mountWidth, mountThickness * 6]);
        
        //4 screws
        translate([structureWidth/4+handlePinSpacingX/2, handlePinSpacingY/2, plateH + nutInsertionHeight/2]) roundedBox([structureWidth,structureLength/2,nutInsertionHeight],3, true);
        translate([-structureWidth/4-handlePinSpacingX/2, handlePinSpacingY/2, plateH + nutInsertionHeight/2]) roundedBox([structureWidth,structureLength/2,nutInsertionHeight],3, true);
        translate([structureWidth/4+handlePinSpacingX/2, -handlePinSpacingY/2, plateH + nutInsertionHeight/2]) roundedBox([structureWidth,structureLength/2,nutInsertionHeight],3, true);
        translate([-structureWidth/4-handlePinSpacingX/2, -handlePinSpacingY/2, plateH + nutInsertionHeight/2]) roundedBox([structureWidth,structureLength/2,nutInsertionHeight],3, true);
        fourHoles (handlePinSpacingX, handlePinSpacingY, handlePinW, 2*(plateH + nutInsertionHeight), standoffFN, 0);
        
        //mecanno holdes
        translate([0, handlePinSpacingY/2 - structureLength, structureHeight - 5/2 ]) meccanoHoles();
    }

    module meccanoHoles() {
        for (i = [1:7]) {
            fourHoles ((2*i+1)*meccanoSpacing, 0, meccanoPinW, plateH*4, standoffFN, plateH);
        }
}
}