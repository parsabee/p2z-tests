#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SMEAR 0.1
#define smear(N) (1 + SMEAR * randn(0,1) * (N))

#define NUM_TRACK_PARAMS 6
#define NUM_HIT_PARAMS 3

#ifndef BSIZE
#define BSIZE 128
#endif

#ifndef NEVTS
#define NEVTS 100
#endif

#ifndef NTRKS
#define NTRKS 9600
#endif

#ifndef NLAYER
#define NLAYER 20
#endif

#ifndef NITER
#define NITER 5
#endif

#define NB (NTRKS / BSIZE)

#define get_track_batch(trks, event_idx, batch_idx) &(trks[(batch_idx) + NB * (event_idx)])

#define get_hit_batch(hits, event_idx, batch_idx, layer_idx) &(hits[layer_idx + (batch_idx*NLAYER) +(event_idx*NLAYER*NB)])

typedef struct pos {
    float x, y, z;
} pos_t;

typedef struct track_params {
    pos_t pos;
    float ipt, phi, theta;
} track_params_t;

typedef struct track {
    track_params_t params;
    float cov[NUM_TRACK_PARAMS][NUM_TRACK_PARAMS];
    int q;
} track_t;

typedef struct track_batch {
    track_t tracks[BSIZE];
} track_batch_t;

typedef struct hit {
    pos_t pos;
    float cov[NUM_HIT_PARAMS][NUM_HIT_PARAMS];
} hit_t;

typedef struct hit_batch {
    hit_t hit[BSIZE];
} hit_batch_t;


static const track_t *get_input_track() {

    static const track_t input_trk =
    {
        {
            -12.806846618652344, -7.723824977874756, 38.13014221191406,
            0.23732035065189902,
            -2.613372802734375,
            0.35594117641448975
        }, /* params */
        {
                {6.290299552347278e-07,  4.1375109560704004e-08, 2.0973730840978533e-07, -2.804026640189443e-06, -2.419662877381737e-07, -7.755406890332818e-07},
                {4.1375109560704004e-08, 7.526661534029699e-07, 1.5431574240665213e-07, 6.219111130687595e-06, 4.3124190760040646e-07, 1.68539375883925e-06},
                {2.0973730840978533e-07, 1.5431574240665213e-07,9.626245400795597e-08, 2.649119409845118e-07, 3.1068903991780678e-09, 6.676875566525437e-08},
                {-2.804026640189443e-06, 6.219111130687595e-06, 2.649119409845118e-07,  0.00253512163402557, 0.000923913115050627, 0.0008420574605423793},
                {-2.419662877381737e-07, 4.3124190760040646e-07,3.1068903991780678e-09, 0.000923913115050627,0.00040678296006807003,7.356584799406111e-05},
                {-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348}
        }, /* cov */
        1 /* q */
    };

    return &input_trk;
}

static const hit_t *get_input_hit() {
    static const hit_t hit =
    {
            {-20.7824649810791, -12.24150276184082, 57.8067626953125}, /* pos */
            {
             {2.545517190810642e-06, -2.6680759219743777e-06, 0.00014160551654640585},
             {-2.6680759219743777e-06,2.8030024168401724e-06, 0.00012282167153898627},
             {0.00014160551654640585, 0.00012282167153898627, 11.385087966918945}
            } /* cov */
    };

    return &hit;
}

float randn(float mu, float sigma) {
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;
    if (call == 1) {
        call = !call;
        return (mu + sigma * (float) X2);
    } do {
        U1 = -1 + ((float) rand () / RAND_MAX) * 2;
        U2 = -1 + ((float) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    call = !call;
    return (mu + sigma * (float) X1);
}

void transpose( int n, float in[n][n], float out[n][n]) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[i][j] = in[j][i];
        }
    }
}

void mat_mult( int n, float A[n][n], float B[n][n], float C[n][n]) {

    float tmp;
    for (int i = 0; i < n; i++) {
        for( int j = 0; j < n; j++) {
            tmp = A[i][j];
            for (int k = 0; k < n; k++) {
                tmp += A[i][k] * B[k][j];
            }
            C[i][j] = tmp;
        }
    }

}

void mult_helix_prop_transp_endcap(int n, float A[n][n], float B[n][n], float C[n][n]) {

    transpose(n, B, C);
    mat_mult(n, A, C, C);
}

track_batch_t *prepare_tracks(const track_t *input_trk) {

    track_batch_t *result = (track_batch_t *)malloc(sizeof(track_batch_t) * NEVTS * NB);

    if (result) {
        for (int ie = 0; ie < NEVTS; ie++) {
            for (int ib = 0; ib < NB; ib++) {
                for (int it = 0; it < BSIZE; it++) {

                    track_t t = result[ib + NB * ie].tracks[it];

                    /// cast parameters to an array of floats for easier initialization
                    float *params = (float *) &(t.params);
                    float *input_params = (float *) &(input_trk->params);
                    for (int ip = 0; ip < NUM_TRACK_PARAMS; ip++) {
                        params[ip] = smear(input_params[ip]);
                    }

                    for (int i = 0; i < NUM_TRACK_PARAMS; i++) {
                        for (int j = 0; j < NUM_TRACK_PARAMS; j++) {
                            t.cov[i][j] = smear(input_trk->cov[i][j]);
                        }
                    }

                    t.q = (int) (input_trk->q - 2 * ceil(-0.5 + (float) rand() / (float) RAND_MAX));
                }
            }
        }
    }
    return result;
}

hit_batch_t* prepare_hits(const hit_t *inputhit) {
    hit_batch_t* result = (hit_batch_t*) malloc(NLAYER * NEVTS * NB * sizeof(hit_batch_t));  //fixme, align?

    if (result) {
        // store in element order for bunches of bsize matrices (a la matriplex)
        for (size_t lay=0;lay<NLAYER;++lay) {
            for (size_t ie=0;ie<NEVTS;++ie) {
                for (size_t ib=0;ib<NB;++ib) {
                    for (size_t it=0;it<BSIZE;++it) {

                        hit_t hit = result[lay+NLAYER*(ib + NB * ie)].hit[it];

                        float *pos = (float *)&(hit.pos);
                        float *input_pos = (float *)&(inputhit->pos);
                        //pos
                        for (size_t ip=0;ip<NUM_HIT_PARAMS;++ip) {
                            pos[ip] = smear(input_pos[ip]);
                        }

                        //cov
                        for (int i = 0;i < NUM_HIT_PARAMS; i++) {
                            for (int j = 0; j < NUM_HIT_PARAMS; j++) {
                                hit.cov[i][j] = smear(inputhit->cov[i][j]);
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

void propagate_to_z(track_batch_t *input_tracks,
                    hit_batch_t *input_hits,
                    track_batch_t *output_tracks)
{
    const float kfact = (float)(100/3.8);
    float error_prop[BSIZE][NUM_TRACK_PARAMS][NUM_TRACK_PARAMS];
    float temp[BSIZE][NUM_TRACK_PARAMS][NUM_TRACK_PARAMS];

    for (int it = 0; it < BSIZE; it++) {
        hit_t *in_hit = &(input_hits->hit[it]);
        track_t *in_track = &(input_tracks->tracks[it]);
        track_t *out_track = &(output_tracks->tracks[it]);

        const float zout = in_hit->pos.z;
        const float k = (float)(in_track->q) * kfact;
        const float delta_z = zout - in_track->params.pos.z;
        const float pt = 1.f/(in_track->params.ipt);
        const float cos_p = cosf(in_track->params.phi);
        const float sin_p = sinf(in_track->params.phi);
        const float cos_t = cosf(in_track->params.theta);
        const float sin_t = sinf(in_track->params.theta);
        const float pxin = cos_p * pt;
        const float pyin = sin_p * pt;
        const float icos_t = 1.f/cos_t;
        const float icos_tk = icos_t / k;
        const float alpha = delta_z * sin_t * in_track->params.ipt * icos_tk;
        const float sina = sinf(alpha);
        const float cosa = cosf(alpha);

        const track_params_t out_params = {
                in_track->params.pos.x + k * (pxin * sina - pyin * (1.f - cosa)),
                in_track->params.pos.y + k * (pyin * sina - pxin * (1.f - cosa)),
                zout,
                in_track->params.ipt,
                in_track->params.phi + alpha,
                in_track->params.theta
        };

        out_track->params = out_params;

        const float s_cosp_sina = sinf(cos_p * sina);
        const float c_cosp_sina = cosf(cos_p * sina);

        for (int i = 0; i < 6; i++) error_prop[it][i][i] = 1.f;

        error_prop[it][0][2] = cos_p * sin_t * (sin_p * cosa * s_cosp_sina - cosa) * icos_t;
        error_prop[it][0][3] = cos_p * sin_t * delta_z * cosa * (1. - sin_p * s_cosp_sina) * (icos_t * pt) - k * (cos_p * sina - sin_p * (1.f - c_cosp_sina)) * (pt * pt);
        error_prop[it][0][4] = (k*pt)*(-sin_p*sina+sin_p*sin_p*sina*s_cosp_sina-cos_p*(1.-c_cosp_sina));
        error_prop[it][0][5] = cos_p*delta_z*cosa*(1.-sin_p*s_cosp_sina)*(icos_t*icos_t);
        error_prop[it][1][2] = cosa*sin_t*(cos_p*cos_p*s_cosp_sina-sin_p)*icos_t;
        error_prop[it][1][3] = sin_t*delta_z*cosa*(cos_p*cos_p*s_cosp_sina+sin_p)*(icos_t*pt)-k*(sin_p*sina+cos_p*(1.-c_cosp_sina))*(pt*pt);
        error_prop[it][1][4] = (k*pt)*(-sin_p*(1.-c_cosp_sina)-sin_p*cos_p*sina*s_cosp_sina+cos_p*sina);
        error_prop[it][1][5] = delta_z*cosa*(cos_p*cos_p*s_cosp_sina+sin_p)*(icos_t*icos_t);
        error_prop[it][4][2] = -in_track->params.ipt * sin_t * (icos_tk);
        error_prop[it][4][3] = sin_t*delta_z*(icos_tk);
        error_prop[it][4][5] = in_track->params.ipt * delta_z * (icos_t * icos_tk);

        mat_mult(NUM_TRACK_PARAMS, error_prop[it], in_track->cov, temp[it]);
        mult_helix_prop_transp_endcap(NUM_TRACK_PARAMS, error_prop[it], temp[it], out_track->cov);
    }
}

int main() {

    const track_t *inputtrk = get_input_track();
    const hit_t *inputhit = get_input_hit();

    printf("track in pos: %f, %f, %f \n", inputtrk->params.pos.x, inputtrk->params.pos.y, inputtrk->params.pos.z);
    printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk->cov[0][0], inputtrk->cov[1][1], inputtrk->cov[2][2]);
    printf("hit in pos: %f %f %f \n", inputhit->pos.x, inputhit->pos.y, inputhit->pos.z);

    track_batch_t *tracks = prepare_tracks(inputtrk);
    hit_batch_t *hits = prepare_hits(inputhit);
    track_batch_t *outtracks = (track_batch_t *)malloc(sizeof(track_batch_t) * NEVTS * NB);

    printf("done preparing!\n");

    for (int itr = 0; itr < NITER; itr++) {
        for (int ie = 0; ie < NEVTS; ie++) {
            for (int ib = 0; ib < NB; ib++) {
                track_batch_t *btracks = get_track_batch(tracks, ie, ib);
                track_batch_t *obtracks = get_track_batch(outtracks, ie, ib);
                for (int layer = 0; layer < NLAYER; layer++) {
                    hit_batch_t *bhits = get_hit_batch(hits, ie, ib, layer);
                    propagate_to_z(btracks, bhits, obtracks);
                }
            }
        }
    }

    free(tracks);
    free(hits);
    free(outtracks);
}
