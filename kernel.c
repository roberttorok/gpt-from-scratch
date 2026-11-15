#include <math.h>

void mat_mul(
    float *a_data, int a_rows, int a_cols, int a_offset, int a_row_stride, int a_col_stride,
    float *b_data, int b_rows, int b_cols, int b_offset, int b_row_stride, int b_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < a_rows; i++)
    {
        for (int j = 0; j < b_cols; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < a_cols; k++)
            {
                int a_idx = (a_offset + i * a_row_stride + k * a_col_stride);
                int b_idx = (b_offset + k * b_row_stride + j * b_col_stride);
                sum = sum + *(a_data + a_idx) * *(b_data + b_idx);
            }
            int dst_idx = dst_offset + (i * dst_row_stride) + (j * dst_col_stride);
            *(dst_data + dst_idx) = sum;
        }
    }
}

void mat_mul_add(
    float *a_data, int a_rows, int a_cols, int a_offset, int a_row_stride, int a_col_stride,
    float *b_data, int b_rows, int b_cols, int b_offset, int b_row_stride, int b_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < a_rows; i++)
    {
        for (int j = 0; j < b_cols; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < a_cols; k++)
            {
                int a_idx = (a_offset + i * a_row_stride + k * a_col_stride);
                int b_idx = (b_offset + k * b_row_stride + j * b_col_stride);
                sum = sum + *(a_data + a_idx) * *(b_data + b_idx);
            }
            int dst_idx = dst_offset + (i * dst_row_stride) + (j * dst_col_stride);
            *(dst_data + dst_idx) += sum;
        }
    }
}

void scale(
    float *src_data, int rows, int cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride,
    float value)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(dst_data + dst_idx) = *(src_data + src_idx) * value;
        }
    }
}

void scale_bw(
    float *src_grad, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_grad, int dst_offset, int dst_row_stride, int dst_col_stride,
    float value)
{
    for (int i = 0; i < src_rows; i++)
    {
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(src_grad + src_idx) += *(dst_grad + dst_idx) * value;
        }
    }
}

void softmax(
    float *src_data, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        float sum = 0.0;
        float max = -INFINITY;

        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            float val = *(src_data + src_idx);
            if (val > max)
            {
                max = val;
            }
        }

        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            float val = *(src_data + src_idx);
            sum += expf(val - max);
        }

        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(dst_data + dst_idx) = expf(*(src_data + src_idx) - max) / sum;
        }
    }
}

void softmax_bw(
    float *y_data, float *y_grad, int rows, int cols, int y_offset, int y_row_stride, int y_col_stride, 
    float *src_grad, int src_offset, int src_row_stride, int src_col_stride)
{
    for (int i = 0; i < rows; i++)
    {
        float dot = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            int y_id = y_offset + (y_row_stride * i) + (j * y_col_stride);
            dot += y_grad[y_id] * y_data[y_id];
        }

        for (int j = 0; j < cols; j++)
        {
            int y_id = y_offset + (y_row_stride * i) + (j * y_col_stride);
            int src_id = src_offset + (src_row_stride * i) + (j * src_col_stride);
            src_grad[src_id] += y_data[y_id] * (y_grad[y_id] - dot);
        }
    }
}

void casual_mask(
    float *src_data, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(dst_data + dst_idx) = j <= i ? *(src_data + src_idx) : -INFINITY;
        }
    }
}

void casual_mask_bw(
    float *src_data, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            if (j <= i)
            {
                *(src_data + src_idx) += *(dst_data + dst_idx);
            }
        }
    }
}

void add(
    float *a_data, int a_rows, int a_cols, int a_offset, int a_row_stride, int a_col_stride,
    float *b_data, int b_rows, int b_cols, int b_offset, int b_row_stride, int b_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < a_rows; i++)
    {
        for (int j = 0; j < a_cols; j++)
        {
            int a_idx = a_offset + (a_row_stride * i) + (j * a_col_stride);
            int b_idx = b_offset + (b_row_stride * i) + (j * b_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(dst_data + dst_idx) = *(a_data + a_idx) + *(b_data + b_idx);
        }
    }
}


void add_bw(
    float *src_data, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(src_data + src_idx) += *(dst_data + dst_idx);
        }
    }
}

void relu(
    float *src_data, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(dst_data + dst_idx) = *(src_data + src_idx) >= 0.0 ? *(src_data + src_idx) : 0.0;
        }
    }
}

void relu_bw(
    float *src_data, float *src_grad, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_grad, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(src_grad + src_idx) += *(src_data + src_idx) >= 0.0 ? *(dst_grad + dst_idx) : 0.0;
        }
    }
}

void rms_norm(
    float *src_data, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        float ms = 0.0;
        for (int j = 0; j < src_cols; j++)
        {
            int idx = src_offset + (src_row_stride * i) + (src_col_stride * j);
            ms += *(src_data + idx) * *(src_data + idx);
        }
        ms /= src_cols;
        float rms = sqrtf(ms + 1e-5f);
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (src_col_stride * j);
            int dst_idx = dst_offset + (dst_row_stride * i) + (dst_col_stride * j);
            *(dst_data + dst_idx) = *(src_data + src_idx) / rms; // no g on purpose
        }
    }
}

void rms_norm_bw(
    float *src_data, float *src_grad, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_grad, int dst_offset, int dst_row_stride, int dst_col_stride)
{
    for (int i = 0; i < src_rows; i++)
    {
        // pass 1: recompute rms, and dot = sum_j dy[j] * x[j]
        float ms = 0.0f;
        float dot = 0.0f;
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (src_col_stride * j);
            int dst_idx = dst_offset + (dst_row_stride * i) + (dst_col_stride * j);
            float x = src_data[src_idx];
            ms += x * x;
            dot += dst_grad[dst_idx] * x;
        }
        ms /= src_cols;
        float rms = sqrtf(ms + 1e-5f); // must match eps in rms_norm
        float shared = dot / (src_cols * rms * rms * rms);

        // pass 2: direct path minus shared path
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (src_col_stride * j);
            int dst_idx = dst_offset + (dst_row_stride * i) + (dst_col_stride * j);
            src_grad[src_idx] += dst_grad[dst_idx] / rms - src_data[src_idx] * shared;
        }
    }
}

void cross_entropy(
    float *src_data, int src_rows, int src_cols, int src_offset, int src_row_stride, int src_col_stride,
    float *dst_data, int dst_offset, int dst_row_stride, int dst_col_stride,
    int *targets,
    float *l)
{
    float sum_loss = 0.0;
    for (int i = 0; i < src_rows; i++)
    {
        float sum = 0.0;
        float max = -INFINITY;
        for (int j = 0; j < src_cols; j++)
        {
            int source_id = src_offset + (src_row_stride * i) + (j * src_col_stride);
            float val = *(src_data + source_id);
            if (val > max)
            {
                max = val;
            }
        }
        for (int j = 0; j < src_cols; j++)
        {
            int source_id = src_offset + (src_row_stride * i) + (j * src_col_stride);
            float val = *(src_data + source_id);
            sum += expf(val - max);
        }
        for (int j = 0; j < src_cols; j++)
        {
            int src_idx = src_offset + (src_row_stride * i) + (j * src_col_stride);
            int dst_idx = dst_offset + (dst_row_stride * i) + (j * dst_col_stride);
            *(dst_data + dst_idx) = expf(*(src_data + src_idx) - max) / sum;
        }
        int idx = dst_offset + (dst_row_stride * i) + (dst_col_stride * *(targets + i));
        sum_loss += -logf(*(dst_data + idx));
    }
    *l = sum_loss / src_rows;
}

void cross_entropy_bw(
    float *buf_data, int rows, int cols, int buf_offset, int buf_row_stride, int buf_col_stride,
    int *targets,
    float *src_grad, int src_offset, int src_row_stride, int src_col_stride,
    float scale)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int buf_id = buf_offset + (buf_row_stride * i) + (j * buf_col_stride);
            int src_id = src_offset + (src_row_stride * i) + (j * src_col_stride);
            float p = buf_data[buf_id];
            if (j == targets[i])
            {
                p -= 1.0f;
            }
            src_grad[src_id] += p * scale;
        }
    }
}

void embed(
    float *embed_data, int embed_rows, int embed_cols, int embed_offsets, int embed_row_stride, int embed_col_stride,
    float *dst_data, int dst_rows, int dst_cols, int dst_offsets, int dst_row_stride, int dst_col_stride,
    int *ids)
{
    for (int i = 0; i < dst_rows; i++)
    {
        int index_to_copy = *(ids + i);
        for (int j = 0; j < dst_cols; j++)
        {
            int embed_index = embed_offsets + (embed_row_stride * index_to_copy) + (embed_col_stride * j);
            int dst_idx = dst_offsets + (dst_row_stride * i) + (dst_col_stride * j);
            *(dst_data + dst_idx) = *(embed_data + embed_index);
        }
    }
}

// backward of embed: send each output row's grad back to the table row it was copied from
// table.grad[ids[i]] += out.grad[i]   (+= so repeated ids accumulate)
void embed_bw(
    float *embed_grad, int embed_rows, int embed_cols, int embed_offset, int embed_row_stride, int embed_col_stride,
    float *dest_grad, int dest_rows, int dest_cols, int dest_offset, int dest_row_stride, int dest_col_stride,
    int *ids)
{
    for (int i = 0; i < dest_rows; i++)
    {
        int row = ids[i];
        for (int j = 0; j < dest_cols; j++)
        {
            int embed_id = embed_offset + (embed_row_stride * row) + (embed_col_stride * j);
            int dest_id = dest_offset + (dest_row_stride * i) + (dest_col_stride * j);
            embed_grad[embed_id] += dest_grad[dest_id];
        }
    }
}

void learn_simple(
    float *data,
    float *grad,
    int rows,
    int cols,
    int offset,
    int row_stride,
    int col_stride,
    float lr)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int index = offset + (i * row_stride) + (j * col_stride);
            *(data + index) = *(data + index) - (lr * *(grad + index));
            *(grad + index) = 0.0;
        }
    }
}
