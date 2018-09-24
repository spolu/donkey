// A Raspberry Pi camera mount.

$fs = .2;
$fa = 5;

// Dimensions of the camera board.
cameraWidth = 25;
cameraHeight = 24;
aperatureWidth = 8;
aperatureYOffset = 5.5;
cameraBoardThickness = 1;

holeYOffset = 9.5;
holeXSpacing = 21;
holeYSpacing = 12.5;
holeDiameter = 2;

standoffDiameter = 6;
standoffHeight = 3;

mountMargin = 3;
mountBottomMargin = 10; // To accommodate the cable.
mountWidth = cameraWidth + 2*mountMargin;
mountHeight = cameraHeight + mountMargin + mountBottomMargin;

mountThickness = 3;

baseWidth = 15;
mountingHoleSpacing = 15;
mountingHoleDiameter = 3;

aperatureMargin = 1;

maxModelSize = 30;

rotate([0,-21.5,0])
camMount();

module camMount() {
    union() {
 rotate([0,15,0]) translate([-54,0,6.5])mountBase();
mainMount();
cameraBoard();
}}

// The camera board.
module cameraBoard(){
%translate([mountBottomMargin, -cameraWidth/2, mountThickness+standoffHeight])
cube(size=[cameraHeight, cameraWidth, cameraBoardThickness]);
}

// The mount base.
module mountBase() {
  union() {
    difference(){
    // connection between cam mount and base mount    
    translate([-3, -mountWidth/2, -.1])
  rotate([0,6.35,0])cube(size=[cameraHeight + mountMargin + mountBottomMargin+20.5,
             cameraWidth + 2*mountMargin,
         mountThickness]);
 
  // CSI cable slot
  extra=2;
  translate([25,15.5,0])rotate([30,0,90])rPiCSI(extra,extra);
        }
//screw mount
rotate([0, -90, 0])
difference() {
  translate([0, -mountWidth/2, 0])
  cube(size=[baseWidth + mountThickness-8.0, mountWidth, mountThickness]);
  
  translate([baseWidth/2 + mountThickness-3.5, -1-mountingHoleSpacing/2, 0])
  cylinder(d=mountingHoleDiameter, h=2*maxModelSize, center=true);
  
  translate([baseWidth/2 + mountThickness-3.5, mountingHoleSpacing/2, 0])
  cylinder(d=mountingHoleDiameter, h=2*maxModelSize, center=true);
}
}
}
// The main part of the mount.
module mainMount() {
difference() {
  union() {
  translate([0, -mountWidth/2, 0])
  cube(size=[cameraHeight + mountMargin + mountBottomMargin,
             cameraWidth + 2*mountMargin,
             mountThickness]);
  // mount standoffs
  translate([holeYOffset + mountBottomMargin, -holeXSpacing/2, mountThickness+standoffHeight/2])
  cylinder(d=standoffDiameter, h=standoffHeight, center=true);

  translate([holeYOffset + mountBottomMargin, holeXSpacing/2, mountThickness+standoffHeight/2])
  cylinder(d=standoffDiameter, h=standoffHeight, center=true);
  
  translate([holeYOffset + holeYSpacing + mountBottomMargin, -holeXSpacing/2, mountThickness+standoffHeight/2])
  cylinder(d=standoffDiameter, h=standoffHeight, center=true);
  
  translate([holeYOffset + holeYSpacing + mountBottomMargin, holeXSpacing/2, mountThickness+standoffHeight/2])
  cylinder(d=standoffDiameter, h=standoffHeight, center=true);
  }
  
  // mounting holes
  translate([holeYOffset + mountBottomMargin, -holeXSpacing/2, 0])
  cylinder(d=holeDiameter, h=2*maxModelSize, center=true);
  
  translate([holeYOffset + mountBottomMargin, holeXSpacing/2, 0])
  cylinder(d=holeDiameter, h=2*maxModelSize, center=true);
  
  translate([holeYOffset + holeYSpacing + mountBottomMargin, -holeXSpacing/2, 0])
  cylinder(d=holeDiameter, h=2*maxModelSize, center=true);
  
  translate([holeYOffset + holeYSpacing + mountBottomMargin, holeXSpacing/2, 0])
  cylinder(d=holeDiameter, h=2*maxModelSize, center=true);
  
  translate([mountBottomMargin + aperatureYOffset,
              -aperatureWidth/2 - aperatureMargin,
              -maxModelSize])
  cube(size=[aperatureWidth + 2*aperatureMargin,
             aperatureWidth + 2*aperatureMargin,
             2*maxModelSize]);
}
}
module rPiCSI (extra=0,addedW=0) {
    translate([-26.7,-20,0])
      rotate([0,0,0])
        cube([20+extra,0.5+addedW,50]);
}