#ifdef __cplusplus
extern "C" {
#endif

int init(int num);
int process_cpu(char* input, int input_size, char* output, int output_size);
int process_gpu(char* input, int input_size, char* output, int output_size);
int release();

#ifdef __cplusplus
}
#endif
