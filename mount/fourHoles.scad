
module fourMounts (spacingX=10, spacingY=10, mountDiameter=2.7*2, pinHeight=platePinH, standoffFN=30, plateH=5) {
    $fn=standoffFN;
    translate([spacingX/2, spacingY/2, plateH + pinHeight/2]) cylinder(r=mountDiameter/2,h=pinHeight,center=true);
    translate([-spacingX/2,spacingY/2,plateH  + pinHeight/2]) cylinder(r=mountDiameter/2,h=pinHeight,center=true);
    translate([spacingX/2,-spacingY/2,plateH  + pinHeight/2]) cylinder(r=mountDiameter/2,h=pinHeight,center=true);
    translate([-spacingX/2,-spacingY/2,plateH  + pinHeight/2]) cylinder(r=mountDiameter/2,h=pinHeight,center=true);    
}

module fourHoles (spacingX=10, spacingY=10, pinDiameter=2.7, holeHeight=100, standoffFN=30, plateH) {
    $fn=standoffFN;
    translate([spacingX/2, spacingY/2, plateH/2]) cylinder(r=pinDiameter/2,h=holeHeight,center=true);
    translate([-spacingX/2,spacingY/2,plateH/2]) cylinder(r=pinDiameter/2,h=holeHeight,center=true);
    translate([spacingX/2,-spacingY/2,plateH/2]) cylinder(r=pinDiameter/2,h=holeHeight,center=true);
    translate([-spacingX/2,-spacingY/2,plateH/2]) cylinder(r=pinDiameter/2,h=holeHeight,center=true);    
}
