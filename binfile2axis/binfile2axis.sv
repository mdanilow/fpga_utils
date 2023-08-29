`timescale 1ns / 1ps

module binfile2axis #(
    parameter IMG_PATH = "",
    parameter height = 224,
    parameter width = 224,
    parameter FOLDING = 1,
    parameter channels = 3,
    parameter datawidth = 8,
    parameter TO_SEND = 1,

    localparam bytes_per_num = datawidth / 8,
    localparam total_bytes = height * width * channels * bytes_per_num,
    localparam PARALLEL_BYTES = (channels / FOLDING) * bytes_per_num,
    localparam BYTES_CNT_WIDTH = $clog2(total_bytes) + 1
)
(
  //hdmi outputs
  output reg clk,
  
  //master axis interface
  output reg [(PARALLEL_BYTES * 8) - 1 : 0] m_axis_0_tdata = 0,
  output reg m_axis_0_tvalid = 0,
  input m_axis_0_tready,
  output reg [BYTES_CNT_WIDTH - 1 : 0] bytes_cnt = 0
); 


initial
begin
  while(1)
  begin
    #1 clk=1'b0;
	#1 clk=1'b1;
  end
end  

reg read_file = 1;
integer file;
integer sent = 0;
string file_idx_s;
string filepath = IMG_PATH;
int len_filepath = filepath.len();


always @(posedge clk)
begin
    if(read_file == 0 && m_axis_0_tready)
    begin
        for(integer i = 0; i < PARALLEL_BYTES; i = i + 1)
            m_axis_0_tdata[(PARALLEL_BYTES-i)*8 - 1 -: 8] <= $fgetc(file);
        bytes_cnt <= bytes_cnt + PARALLEL_BYTES;
        m_axis_0_tvalid <= 1; 

      if(bytes_cnt >= total_bytes)
      begin
          sent <= sent + 1;
          bytes_cnt <= 0;
          m_axis_0_tvalid <= 0;
          read_file <= 1;
      end
    end
end


always @(posedge clk)
begin
    if(read_file == 1 && sent < TO_SEND)
    begin  

        if(TO_SEND > 1)
        begin
          $sformat(file_idx_s, "%0d", sent);
          filepath = {filepath.substr(0, len_filepath-6), file_idx_s, ".bin"};
        end

        $display("FILEPATH %s", filepath, file_idx_s.len());
        file = $fopen(filepath,"rb");
        
        read_file <= 0;
    end
end

// assign m_axis_0_tvalid = !skip_header;

endmodule

