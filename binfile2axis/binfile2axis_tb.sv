`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/20/2022 10:45:00 AM
// Design Name: 
// Module Name: xcorr_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module binfile2axis_tb(

);

localparam in_features_height = 224;
localparam in_features_width = 224;
localparam channels = 3;
localparam folding = 1;
localparam datawidth = 8;

localparam channels_per_packet = (channels / folding);
localparam packet_width = channels_per_packet * datawidth;
localparam FOLD_CNT_WIDTH = $clog2(folding);
localparam X_CNT_WIDTH = $clog2(in_features_width);
localparam Y_CNT_WIDTH = $clog2(in_features_height);


wire clk;
wire [packet_width - 1 : 0] features_m_axis_tdata;
wire features_m_axis_tvalid;
wire features_m_axis_tready;
wire [18 : 0] bytes_cnt;
wire signed [0 : channels_per_packet - 1][datawidth - 1 : 0] unpacked_features;

reg resetn = 0;
reg [X_CNT_WIDTH - 1 : 0] x_cnt = 0;
reg [Y_CNT_WIDTH - 1 : 0] y_cnt = 0;


initial
begin
    #10
    resetn <= 1;
end


binfile2axis #(
    .IMG_PATH("/home/vision/danilowi/fpga_utils/binfile2axis/bytearray.bin"),
    .FOLDING(folding),
    .height(in_features_height),
    .width(in_features_width),
    .channels(channels),
    .datawidth(datawidth),
    .TO_SEND(1)
)
file_input (
    .clk(clk),
    .m_axis_0_tdata(features_m_axis_tdata),
    .m_axis_0_tvalid(features_m_axis_tvalid),
    .m_axis_0_tready(features_m_axis_tready),
    .bytes_cnt(bytes_cnt)
);
assign features_m_axis_tready = 1'b1;


always @(posedge clk)
begin
    if(features_m_axis_tvalid && features_m_axis_tready)
        x_cnt <= x_cnt + 1;

    if(x_cnt == in_features_width - 1)
    begin
        x_cnt <= 0;
        y_cnt <= y_cnt + 1;

        if(y_cnt == in_features_height - 1)
            y_cnt <= 0;
    end
end


// unpack_output #(
//     .IN_WIDTH(packet_width),
//     .OUT_WIDTH(datawidth)
// )
// unpackfinn(
//     .in(features_m_axis_tdata),

//     .out(unpacked_features)
// );


endmodule
